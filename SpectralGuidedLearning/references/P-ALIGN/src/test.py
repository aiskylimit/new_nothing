import argparse
import __future__
import importlib.abc
import importlib.machinery
import json
import os
import sys

import jsonlines
from transformers import AutoTokenizer
from tqdm import tqdm


class _PostponedAnnotationsLoader(importlib.abc.Loader):
    """Compile one dependency module with postponed annotations."""

    def __init__(self, fullname, delegate):
        self.fullname = fullname
        self.delegate = delegate

    def create_module(self, spec):
        create_module = getattr(self.delegate, "create_module", None)
        return create_module(spec) if create_module else None

    def exec_module(self, module):
        get_source = getattr(self.delegate, "get_source", None)
        source = get_source(self.fullname) if get_source else None
        if source is None or "array.array[int]" not in source:
            self.delegate.exec_module(module)
            return

        get_filename = getattr(self.delegate, "get_filename", None)
        filename = get_filename(self.fullname) if get_filename else module.__spec__.origin
        code = compile(
            source,
            filename,
            "exec",
            flags=__future__.annotations.compiler_flag,
            dont_inherit=True,
        )
        exec(code, module.__dict__)


class _FlashInferCompatFinder(importlib.abc.MetaPathFinder):
    """Work around flashinfer 0.6.16's Python 3.11 annotation bug."""

    _TARGET = "flashinfer.comm.fd_exchange"

    def find_spec(self, fullname, path=None, target=None):
        if fullname != self._TARGET:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is not None and spec.loader is not None:
            spec.loader = _PostponedAnnotationsLoader(fullname, spec.loader)
        return spec


def _install_flashinfer_py311_compat():
    if sys.version_info >= (3, 12):
        return
    if not any(isinstance(finder, _FlashInferCompatFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _FlashInferCompatFinder())


_install_flashinfer_py311_compat()

try:
    from vllm import LLM, SamplingParams
except ImportError as e:
    LLM = None
    SamplingParams = None
    _VLLM_IMPORT_ERROR = e
else:
    _VLLM_IMPORT_ERROR = None
    if sys.version_info < (3, 12):
        try:
            import flashinfer.comm  # noqa: F401
        except ImportError:
            # FlashInfer is optional when vLLM uses the FlashAttention backend.
            pass


def _make_llm(model, max_model_len, gpu_memory_utilization=0.8):
    """Create a vLLM instance using FlashAttention."""
    kwargs = dict(
        model=model,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        trust_remote_code=True,
        tensor_parallel_size=1,
    )
    if sys.version_info < (3, 12):
        kwargs["enforce_eager"] = True
    try:
        return LLM(**kwargs, attention_backend="FLASH_ATTN")
    except TypeError as exc:
        if "attention_backend" not in str(exc):
            raise
        os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
        return LLM(**kwargs)


def split_list(lst, batch_size):
    return [lst[i:i + batch_size] for i in range(0, len(lst), batch_size)]


EMPTY_THINK = "<think>\n\n</think>\n\n"


def apply_chat(tokenizer, messages, force_empty_think=False):
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    try:
        text = tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        text = tokenizer.apply_chat_template(messages, **kwargs)
    if text.endswith(EMPTY_THINK):
        return text
    # Templates without enable_thinking (e.g. DeepSeek-R1-Distill) open "<think>\n" in the
    # generation prompt; close it empty so the model answers directly (non-thinking mode).
    stripped = text.rstrip()
    if stripped.endswith("<think>"):
        return stripped[: -len("<think>")] + EMPTY_THINK
    if force_empty_think:
        return text + EMPTY_THINK
    return text


def process_data(json_filename, file_name, llm, batch_size, tokenizer, sampling_params, force_empty_think=False):
    data = []
    with jsonlines.open(json_filename) as infile:
        for item in infile:
            problem_key = next((k for k in ["problem", "question", "input", "content"] if k in item), None)
            answer_key = next((k for k in ["answer", "target", "solution", "ground_truth"] if k in item), None)
            if not problem_key or not answer_key:
                continue
            data.append({
                "prompt_ori": f"Please reason step by step, and put your final answer within \\boxed{{}}.{item[problem_key]}",
                "answer": item[answer_key],
            })
    texts = [
        apply_chat(tokenizer, [{"role": "user", "content": d["prompt_ori"]}], force_empty_think)
        for d in data
    ]
    # The rendered template already carries BOS where the model needs it (DeepSeek);
    # pass token ids so vLLM does not prepend a second one.
    prompts = [{"prompt_token_ids": tokenizer.encode(t, add_special_tokens=False)} for t in texts]
    out_dir = os.path.dirname(file_name)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    results = []
    for batch in tqdm(split_list(prompts, batch_size), desc=json_filename):
        for output in llm.generate(batch, sampling_params):
            results.append([o.text for o in output.outputs])
    with open(file_name, "w", encoding="utf-8") as f:
        for result, item in zip(results, data):
            item["output"] = result
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--input_files", nargs="+", required=True)
    p.add_argument("--output_files", nargs="+", required=True)
    p.add_argument("--batch_size", type=int, default=1000)
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--repetition_penalty", type=float, default=1.05)
    p.add_argument("--max_tokens", type=int, default=4096)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.8,
                   help="fraction of total GPU memory vLLM may take; lower it when the GPU is shared")
    p.add_argument("--force_empty_think", action="store_true",
                   help="append an empty <think></think> block even if the chat template does not open one")
    args = p.parse_args()
    if len(args.input_files) != len(args.output_files):
        raise ValueError("input_files and output_files length must match")
    if LLM is None or SamplingParams is None:
        raise ImportError(f"vLLM is required for eval: {_VLLM_IMPORT_ERROR}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    sampling_params = SamplingParams(
        n=args.n, temperature=args.temperature, top_p=args.top_p,
        repetition_penalty=args.repetition_penalty, max_tokens=args.max_tokens,
    )
    llm = _make_llm(args.model, args.max_tokens, args.gpu_memory_utilization)
    for inp, out in zip(args.input_files, args.output_files):
        process_data(inp, out, llm, args.batch_size, tokenizer, sampling_params, args.force_empty_think)


if __name__ == "__main__":
    main()
