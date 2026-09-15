import argparse
import json
import os
import sys

import jsonlines
from transformers import AutoTokenizer
from tqdm import tqdm

try:
    from vllm import LLM, SamplingParams
except ImportError as e:
    LLM = None
    SamplingParams = None
    _VLLM_IMPORT_ERROR = e
else:
    _VLLM_IMPORT_ERROR = None


def _make_llm(model, max_model_len):
    """vLLM 0.27 + FLASHINFER on Python 3.11 imports flashinfer.comm, which
    uses `array.array[int]` and raises TypeError. Prefer FLASH_ATTN; skip
    torch.compile on <3.12 so that import is never reached."""
    kwargs = dict(
        model=model,
        gpu_memory_utilization=0.8,
        max_model_len=max_model_len,
        trust_remote_code=True,
        tensor_parallel_size=1,
    )
    if sys.version_info < (3, 12):
        kwargs["enforce_eager"] = True
    try:
        return LLM(**kwargs, attention_backend="FLASH_ATTN")
    except TypeError:
        os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
        return LLM(**kwargs)


def split_list(lst, batch_size):
    return [lst[i:i + batch_size] for i in range(0, len(lst), batch_size)]


def apply_chat(tokenizer, messages):
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def process_data(json_filename, file_name, llm, batch_size, tokenizer, sampling_params):
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
    texts = [apply_chat(tokenizer, [{"role": "user", "content": d["prompt_ori"]}]) for d in data]
    out_dir = os.path.dirname(file_name)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    results = []
    for batch in tqdm(split_list(texts, batch_size), desc=json_filename):
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
    llm = _make_llm(args.model, args.max_tokens)
    for inp, out in zip(args.input_files, args.output_files):
        process_data(inp, out, llm, args.batch_size, tokenizer, sampling_params)


if __name__ == "__main__":
    main()
