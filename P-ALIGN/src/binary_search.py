from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import json
import os
import time
from tqdm import tqdm


model = None
tokenizer = None
MAX_NEW_TOKENS = 256


def apply_chat(messages):
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def chat(prompt, model):
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = apply_chat(messages)
    device = getattr(model, "device", None) or next(model.parameters()).device
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=MAX_NEW_TOKENS
    )

    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(
        generated_ids, skip_special_tokens=True
    )[0]
    return response


def split_sentences(text):
    """
    Simple sentence-level splitter.
    You can replace this with a more robust one if needed.
    """
    sentences = text.split(". ")
    return [
        s.strip() + "." if not s.endswith(".") else s.strip()
        for s in sentences if s.strip()
    ]


def reasoning_sufficiency_check(question, reasoning_part):
    """
    Check whether the partial reasoning is sufficient.
    Returns:
        response_content (str)
        is_sufficient (bool)
    """

    prompt = f"""
You are a reasoning evaluator.

You are given a partial reasoning prefix extracted from a longer chain-of-thought.
Your task is to judge whether this prefix already contains the essential logical structure and key transformations needed to complete the solution.

- Reply "[ENOUGH]" if the prefix establishes the core reasoning steps such that the remaining reasoning is straightforward or routine.
- Reply "[NOT_ENOUGH]" if any crucial reasoning step is still missing, making it difficult to reliably complete the solution.

Reply with exactly one token: [ENOUGH] or [NOT_ENOUGH].

Question:
{question}

Partial reasoning:
{reasoning_part}
"""

    try:
        response = chat(prompt, model)
        is_sufficient = (
            "[ENOUGH]" in response or response.strip() == "ENOUGH"
        )
        return response, is_sufficient
    except Exception as e:
        print(f"Model error: {e}")
        return f"ERROR: {e}", False


def find_minimal_sufficient_prefix(question, sentences, sleep_sec=0.5):
    """
        sufficient_reasoning (str)
        prefix_len (int)
        is_sufficient (bool)
        best_response (str)
    """

    total = len(sentences)
    left, right = 1, total
    best_idx = None
    best_response = ""

    print("\n" + "=" * 80)
    print("【开始二分查找最短充分前缀】")
    print(f"总句子数：{total}")
    print(f"初始搜索区间：[{left}, {right}]")
    print("=" * 80)

    step = 1
    while left <= right:
        mid = (left + right) // 2
        prefix_text = " ".join(sentences[:mid])

        print(f"\n【第 {step} 轮评估】")
        print(f"当前搜索区间：left={left}, right={right}")
        print(f"检查前缀句子数：{mid}/{total}")

        t0 = time.time()
        response, is_sufficient = reasoning_sufficiency_check(
            question, prefix_text
        )
        dt = time.time() - t0

        print(f"模型输出：{response}")
        print(f"判定结果：{'ENOUGH' if is_sufficient else 'NOT_ENOUGH'}")
        print(f"评估耗时：{dt:.2f} 秒")

        if is_sufficient:
            best_idx = mid
            best_response = response
            print("动作：当前前缀已足够 → 向左尝试更短前缀")
            right = mid - 1
        else:
            print("动作：当前前缀不足 → 向右增加前缀长度")
            left = mid + 1

        step += 1
        time.sleep(sleep_sec)

    print("\n" + "-" * 80)
    print("【二分查找结束】")

    if best_idx is None:
        print("⚠️ 未找到充分前缀，回退为完整推理")
        return " ".join(sentences), total, False, ""
    else:
        print(f"✅ 最短充分前缀长度：{best_idx}/{total}")
        print(f"前缀比例：{best_idx / total:.4f}")
        return " ".join(sentences[:best_idx]), best_idx, True, best_response


def process_jsonl(input_file, output_file, sleep_sec=0.5):
    out_dir = os.path.dirname(output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    open(output_file, "w", encoding="utf-8").close()

    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for idx, line in enumerate(tqdm(lines, desc="处理数据")):
        data = json.loads(line)
        question = data.get("question", "")
        full_reasoning = data.get("Long-CoT", "")

        if not question or not full_reasoning:
            continue

        sentences = split_sentences(full_reasoning)

        prefix_text, prefix_len, ok, eval_resp = find_minimal_sufficient_prefix(
            question, sentences, sleep_sec=sleep_sec
        )

        result = {
            "id": data.get("id", idx),
            "answer": data.get("answer", ""),
            "question": question,
            "sufficient_reasoning": prefix_text,
            "sufficient_sentences": prefix_len,
            "total_sentences": len(sentences),
            "prefix_ratio": prefix_len / len(sentences) if sentences else 0.0,
            "is_sufficient": ok,
            "evaluator_response": eval_resp
        }

        with open(output_file, "a", encoding="utf-8") as out:
            out.write(json.dumps(result, ensure_ascii=False) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Adaptive prefix truncation via binary search")
    parser.add_argument("--model", type=str, required=True, help="HF model id or local path")
    parser.add_argument("--input_file", type=str, required=True, help="JSONL with question + Long-CoT")
    parser.add_argument("--output_file", type=str, required=True, help="JSONL output path")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--sleep_sec", type=float, default=0.5)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    MAX_NEW_TOKENS = args.max_new_tokens
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    process_jsonl(args.input_file, args.output_file, sleep_sec=args.sleep_sec)
    print("\n✅ 全部数据处理完成")
