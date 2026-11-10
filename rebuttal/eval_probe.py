"""GPU-side evaluator check: imports, CUDA kernel, local backbone and zero LoRA."""
import argparse
import importlib.metadata
import importlib.util
import json
import os
import tempfile
from pathlib import Path

from rebuttal.common import ROOT, write


def zero_adapter(config, model, destination):
    """Create a rank-16 q_proj adapter without loading another backbone in RAM."""
    import torch
    from peft import LoraConfig
    from safetensors.torch import save_file
    rank = 16
    hidden = config.hidden_size
    head_dim = getattr(config, 'head_dim', None) or hidden // config.num_attention_heads
    output = config.num_attention_heads * head_dim
    adapter = LoraConfig(r=rank, lora_alpha=rank, target_modules=['q_proj'],
                         task_type='CAUSAL_LM', inference_mode=True,
                         base_model_name_or_path=str(model))
    adapter.save_pretrained(destination)
    weights = {}
    for layer in range(config.num_hidden_layers):
        prefix = f'base_model.model.model.layers.{layer}.self_attn.q_proj'
        weights[prefix+'.lora_A.weight'] = torch.zeros(rank, hidden)
        weights[prefix+'.lora_B.weight'] = torch.zeros(output, rank)
    save_file(weights, str(destination/'adapter_model.safetensors'))
    return json.loads((destination/'adapter_config.json').read_text())


def cuda_runtime():
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Evaluator probe expects one visible, working CUDA GPU')
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('Evaluator GPU must support bf16')
    x = torch.ones((16,16), device='cuda', dtype=torch.bfloat16)
    value = x @ x
    torch.cuda.synchronize()
    if not torch.isfinite(value).all():
        raise RuntimeError('CUDA bf16 matmul failed')
    return {'torch_cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
            'capability': list(torch.cuda.get_device_capability(0))}


def load_backend():
    spec = importlib.util.spec_from_file_location('eval_probe_backend', ROOT/'palign/src/test.py')
    backend = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(backend)
    return backend


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    result = {'status':'failed', 'model':str(args.model), 'scope':'small synthetic zero-LoRA probe; not full-context memory validation'}
    try:
        os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')
        from transformers import AutoConfig, AutoTokenizer
        result.update(cuda_runtime())
        result['versions'] = {name:importlib.metadata.version(name)
                              for name in ('torch','vllm','transformers','peft','math-verify')}
        backend = load_backend()
        if backend.LLM is None:
            raise RuntimeError('Shared vLLM evaluator failed to import') from backend._VLLM_IMPORT_ERROR
        from vllm import SamplingParams
        from vllm.lora.request import LoRARequest
        config = AutoConfig.from_pretrained(str(args.model), local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='eval-zero-lora-', dir=args.out.parent) as folder:
            adapter_dir = Path(folder)
            adapter = zero_adapter(config, args.model, adapter_dir)
            llm = backend._make_llm(str(args.model), 2048, 1, .8, seed=42,
                adapter_config=adapter, enforce_eager=True, max_num_seqs=1)
            prompt = backend.apply_chat(tokenizer, [{'role':'user','content':'What is 1 + 1?'}])
            output = llm.generate([prompt], SamplingParams(n=1, temperature=0, max_tokens=8),
                                 lora_request=LoRARequest('preflight-zero', 1, str(adapter_dir)))
            if len(output) != 1 or len(output[0].outputs) != 1 or not output[0].outputs[0].token_ids:
                raise RuntimeError('vLLM zero-LoRA probe produced no completion tokens')
            result.update(status='passed', generated_tokens=len(output[0].outputs[0].token_ids), lora_rank=16)
    except BaseException as error:
        result['error'] = repr(error)
        raise
    finally:
        write(args.out, result)


if __name__ == '__main__':
    main()
