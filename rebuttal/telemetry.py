"""Per-process counters; FLOPs are an explicit dense-model estimate, not measured."""
import atexit
import importlib.metadata
import os
import threading
import time
from pathlib import Path

from rebuttal.common import write

STATE = {'forward_tokens': 0, 'autograd_tokens': 0, 'sample_presentations': 0,
         'optimization_steps': 0, 'estimated_flops': 0.0, 'installed': False}
LOCAL = threading.local()


def flush(final=False):
    if not os.environ.get('REBUTTAL_STAGE_DIR'):
        return
    import torch
    STATE['finalized'] = bool(final or STATE.get('finalized', False))
    result = {k: v for k, v in STATE.items() if k != 'installed'}
    result.update(pid=os.getpid(), rank=int(os.environ.get('RANK', '0')), unix_time=time.time(),
                  cuda=torch.version.cuda, versions={}, worker_id=os.environ.get('REBUTTAL_WORKER_ID'),
                  distributed_rank=int(os.environ['RANK']) if 'RANK' in os.environ else None)
    for name in ('torch', 'transformers', 'peft', 'accelerate', 'deepspeed', 'vllm', 'unsloth'):
        try:
            result['versions'][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result['versions'][name] = None
    if torch.cuda.is_initialized():
        device = int(os.environ.get('LOCAL_RANK', '0'))
        result.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(device), device=device)
    else:
        result.update(peak_allocated_bytes=None, peak_reserved_bytes=None)
    write(Path(os.environ['REBUTTAL_STAGE_DIR']) / f'process-{os.getpid()}.json', result)


def activate(model):
    if not os.environ.get('REBUTTAL_STAGE_DIR') or getattr(model, '_rebuttal_instrumented', False):
        return model
    import torch
    model._rebuttal_instrumented = True
    parameters = sum(p.numel() for p in model.parameters())

    def before(module, args, kwargs):
        depth = getattr(LOCAL, 'depth', 0)
        LOCAL.depth = depth + 1
        if depth:
            return
        ids = kwargs.get('input_ids', args[0] if args else None)
        if ids is None:
            ids = kwargs.get('inputs_embeds')
        if ids is None or ids.ndim < 2:
            return
        mask = kwargs.get('attention_mask')
        lengths = mask.sum(-1).tolist() if mask is not None and mask.ndim == 2 else [ids.shape[1]] * ids.shape[0]
        tokens = int(sum(lengths))
        gradient = torch.is_grad_enabled()
        STATE['forward_tokens'] += tokens
        STATE['sample_presentations'] += ids.shape[0]
        if gradient:
            STATE['autograd_tokens'] += tokens
        # Frozen-backbone input gradients approximate one extra forward; GC adds
        # recomputation. Attention and adapter overhead are approximation limits.
        multiplier = 1 + int(gradient) + int(gradient and module.training and getattr(module, 'is_gradient_checkpointing', False))
        config = getattr(module, 'config', None)
        h, layers = getattr(config, 'hidden_size', 0), getattr(config, 'num_hidden_layers', 0)
        cache = kwargs.get('past_key_values')
        past = cache.get_seq_length() if cache is not None and hasattr(cache, 'get_seq_length') else 0
        attention = 4.0 * layers * h * sum(length * (length + past) for length in lengths)
        STATE['estimated_flops'] += (2.0 * parameters * tokens + attention) * multiplier

    def after(module, args, kwargs, output):
        LOCAL.depth = max(0, getattr(LOCAL, 'depth', 1) - 1)

    model.register_forward_pre_hook(before, with_kwargs=True)
    model.register_forward_hook(after, with_kwargs=True, always_call=True)
    return model


def install():
    if not os.environ.get('REBUTTAL_STAGE_DIR') or STATE['installed']:
        return
    import torch
    STATE['installed'] = True
    original = torch.optim.Optimizer.__init__

    def optimizer_init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self.register_step_post_hook(lambda *unused: STATE.__setitem__('optimization_steps', STATE['optimization_steps'] + 1))
    torch.optim.Optimizer.__init__ = optimizer_init
    atexit.register(flush, True)

    def periodic():
        while True:
            time.sleep(10)
            try:
                flush()
            except Exception:
                pass
    threading.Thread(target=periodic, daemon=True).start()


def integrated_gradients(model, batch_size, sequence_length, answer_length):
    """Count the direct decoder/head path used by SSFT, without changing its graph."""
    if not os.environ.get('REBUTTAL_STAGE_DIR'):
        return
    install()
    tokens = batch_size * sequence_length
    decoder_parameters = sum(p.numel() for p in model.model.parameters())
    head_parameters = model.lm_head.weight.numel()
    config = model.config
    forward = (2.0 * decoder_parameters * tokens
               + 4.0 * config.num_hidden_layers * config.hidden_size * batch_size * sequence_length ** 2)
    # Input-gradient backward, plus decoder recomputation when checkpointed.
    multiplier = 2 + int(model.model.training and getattr(model, 'is_gradient_checkpointing', False))
    STATE['forward_tokens'] += tokens
    STATE['autograd_tokens'] += tokens
    STATE['sample_presentations'] += batch_size
    STATE['estimated_flops'] += forward * multiplier + 4.0 * head_parameters * batch_size * answer_length


def inference(outputs, config):
    if not os.environ.get('REBUTTAL_STAGE_DIR'):
        return
    install()
    hidden, layers = config.hidden_size, config.num_hidden_layers
    intermediate, vocabulary = config.intermediate_size, config.vocab_size
    kv = getattr(config, 'num_key_value_heads', config.num_attention_heads) * (hidden // config.num_attention_heads)
    n = hidden * vocabulary * (1 if getattr(config, 'tie_word_embeddings', False) else 2)
    n += layers * (2 * hidden * hidden + 2 * hidden * kv + 3 * hidden * intermediate)
    tokens = sum(len(o.prompt_token_ids) + sum(len(c.token_ids) for c in o.outputs) for o in outputs)
    attention = 0.0
    for output in outputs:
        prompt = len(output.prompt_token_ids)
        attention += 4.0 * layers * hidden * (prompt ** 2 + sum(
            len(c.token_ids) * prompt + len(c.token_ids) * (len(c.token_ids)+1) / 2 for c in output.outputs))
    STATE['forward_tokens'] += tokens
    STATE['sample_presentations'] += len(outputs)
    STATE['estimated_flops'] += 2.0 * n * tokens + attention
    flush()
