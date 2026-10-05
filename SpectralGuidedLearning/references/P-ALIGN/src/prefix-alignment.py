import argparse
import json
import os

import jsonlines
from transformers import AutoTokenizer
from tqdm import tqdm
from vllm import LLM, SamplingParams


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
            question = item["question"]
            sufficient_reasoning = item["sufficient_reasoning"]
            prompt = (
                "Please continue from the draft and solve the problem step by step, "
                "and put your final answer within \\boxed{}. "
                "I will provide you with some prior knowledge as a draft to assist you in solving the question."
                f"*Question*:{question}\n"
                f"*Prefix*:{sufficient_reasoning}"
            )
            data.append({
                "question": question,
                "sufficient_reasoning": sufficient_reasoning,
                "prompt": prompt,
            })

    existing = set()
    if os.path.exists(file_name):
        with open(file_name, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    existing.add(json.loads(line)["question"])
                except Exception:
                    continue

    out_dir = os.path.dirname(file_name)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    total_batches = (len(data) + batch_size - 1) // batch_size
    with open(file_name, "a", encoding="utf-8") as file:
        for batch_idx in tqdm(range(total_batches), desc="Generating"):
            start, end = batch_idx * batch_size, (batch_idx + 1) * batch_size
            batch_data = [d for d in data[start:end] if d["question"] not in existing]
            if not batch_data:
                continue
            texts = [apply_chat(tokenizer, [{"role": "user", "content": d["prompt"]}]) for d in batch_data]
            outputs = llm.generate(texts, sampling_params)
            for output, item in zip(outputs, batch_data):
                item["output"] = output.outputs[0].text
                file.write(json.dumps(item, ensure_ascii=False) + "\n")
            file.flush()
            os.fsync(file.fileno())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--input_file", required=True)
    p.add_argument("--output_file", required=True)
    p.add_argument("--batch_size", type=int, default=500)
    p.add_argument("--max_tokens", type=int, default=32768)
    p.add_argument("--max_model_len", type=int, default=None)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top_p", type=float, default=0.9)
    args = p.parse_args()
    max_model_len = args.max_model_len or args.max_tokens
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    sampling_params = SamplingParams(
        n=1, temperature=args.temperature, top_p=args.top_p,
        repetition_penalty=1.05, max_tokens=args.max_tokens,
    )
    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=max_model_len,
        trust_remote_code=True,
        tensor_parallel_size=1,
    )
    process_data(args.input_file, args.output_file, llm, args.batch_size, tokenizer, sampling_params)


if __name__ == "__main__":
    main()
