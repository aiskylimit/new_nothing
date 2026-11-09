import __future__

import argparse
import importlib.abc
import importlib.machinery
import json
import os
import sys

import jsonlines
from tqdm import tqdm
from transformers import AutoTokenizer

PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."


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
        filename = (
            get_filename(self.fullname) if get_filename else module.__spec__.origin
        )
        code = compile(
            source,
            filename,
            "exec",
            flags=__future__.annotations.compiler_flag,
            dont_inherit=True,
        )
        exec(code, module.__dict__)  # noqa: S102 - compatibility import hook


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
except ImportError as error:
    LLM = None
    SamplingParams = None
    _VLLM_IMPORT_ERROR = error
else:
    _VLLM_IMPORT_ERROR = None
    if sys.version_info < (3, 12):
        try:
            import flashinfer.comm  # noqa: F401
        except ImportError:
            pass


def _make_llm(model, max_model_len, tensor_parallel_size, gpu_memory_utilization, seed=42, adapter_config=None, **runtime_options):
    kwargs = {
        "model": model,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_model_len": max_model_len,
        "trust_remote_code": True,
        "tensor_parallel_size": tensor_parallel_size,
        "seed": seed,
    }
    if adapter_config is not None:
        kwargs["enable_lora"] = True
        kwargs["max_lora_rank"] = max(16, int(adapter_config["r"]))
    if sys.version_info < (3, 12):
        kwargs["enforce_eager"] = True
    kwargs.update(runtime_options)
    try:
        return LLM(**kwargs, attention_backend="FLASH_ATTN")
    except TypeError as error:
        if "attention_backend" not in str(error):
            raise
        os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
        return LLM(**kwargs)


def split_list(items, batch_size):
    return [
        items[index : index + batch_size] for index in range(0, len(items), batch_size)
    ]


def apply_chat(tokenizer, messages):
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def process_data(
    json_filename,
    output_filename,
    llm,
    batch_size,
    tokenizer,
    sampling_params,
    expected_n,
    lora_request=None,
):
    data = []
    with jsonlines.open(json_filename) as infile:
        for item in infile:
            problem_key = next(
                (
                    key
                    for key in ["problem", "question", "input", "content"]
                    if key in item
                ),
                None,
            )
            answer_key = next(
                (
                    key
                    for key in ["answer", "target", "solution", "ground_truth"]
                    if key in item
                ),
                None,
            )
            if not problem_key or not answer_key:
                raise ValueError(f"{json_filename}: row lacks a question or answer field")
            data.append(
                {
                    "prompt_ori": f"{PROMPT}{item[problem_key]}",
                    "answer": item[answer_key],
                }
            )
    texts = [
        apply_chat(tokenizer, [{"role": "user", "content": item["prompt_ori"]}])
        for item in data
    ]
    output_dir = os.path.dirname(output_filename)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    results = []
    generation_kwargs = {"lora_request": lora_request} if lora_request is not None else {}
    for batch in tqdm(split_list(texts, batch_size), desc=json_filename):
        for output in llm.generate(batch, sampling_params, **generation_kwargs):
            candidates = [candidate.text for candidate in output.outputs]
            if len(candidates) != expected_n:
                raise RuntimeError(
                    f"{json_filename}: expected {expected_n} candidates, "
                    f"received {len(candidates)}"
                )
            results.append(candidates)
    if len(results) != len(data):
        raise RuntimeError(
            f"{json_filename}: expected generations for {len(data)} prompts, "
            f"received {len(results)}"
        )
    with open(output_filename, "w", encoding="utf-8") as handle:
        for result, item in zip(results, data):
            item["output"] = result
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input_files", nargs="+", required=True)
    parser.add_argument("--output_files", nargs="+", required=True)
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--max_model_len", type=int, default=32768)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    args = parser.parse_args()
    if len(args.input_files) != len(args.output_files):
        raise ValueError("input_files and output_files length must match")
    if args.max_model_len <= args.max_tokens:
        raise ValueError(
            "max_model_len must be greater than max_tokens to leave room for prompts"
        )
    if LLM is None or SamplingParams is None:
        raise ImportError(f"vLLM is required for eval: {_VLLM_IMPORT_ERROR}")
    from pathlib import Path
    adapter_path = Path(args.model) / "adapter_config.json"
    adapter_config = json.loads(adapter_path.read_text()) if adapter_path.is_file() else None
    base_model = adapter_config["base_model_name_or_path"] if adapter_config else args.model
    tokenizer_source = args.model if (Path(args.model) / "tokenizer_config.json").is_file() else base_model
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source, trust_remote_code=True, local_files_only=True
    )
    lora_request = None
    if adapter_config is not None:
        from vllm.lora.request import LoRARequest
        lora_request = LoRARequest("rebuttal", 1, str(Path(args.model).resolve()))
    sampling_params = SamplingParams(
        n=args.n,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        max_tokens=args.max_tokens,
    )
    llm = _make_llm(
        base_model,
        args.max_model_len,
        args.tensor_parallel_size,
        args.gpu_memory_utilization,
        seed=args.seed,
        adapter_config=adapter_config,
    )
    for input_file, output_file in zip(args.input_files, args.output_files):
        process_data(
            input_file,
            output_file,
            llm,
            args.batch_size,
            tokenizer,
            sampling_params,
            args.n,
            lora_request=lora_request,
        )


if __name__ == "__main__":
    main()
