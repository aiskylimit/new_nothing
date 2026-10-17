"""Adaptive prefix truncation (P-ALIGN step 1): for every teacher trace, binary-search the shortest
sentence prefix the student model judges sufficient to finish the solution.

Port of P-ALIGN/src/binary_search.py with the same splitter (`". "`, see
sgl.segment.rules.palign_sentences), judge prompt, verdict rule ("[ENOUGH]" in the reply, or the
reply is exactly "ENOUGH") and search (lo=1, hi=#sentences; sufficient -> try shorter). A trace
with no sufficient prefix keeps all its sentences (is_sufficient = false).

Instead of one search after another, all searches advance in lockstep: every round asks the judge
about the current midpoint of every unfinished trace at once, so the dataset needs about
log2(#sentences) batched calls. Each search still sees exactly the verdicts it would see alone.

    --backend hf    transformers generate(), one prompt at a time, model generation defaults,
                    256 new tokens: upstream's exact decoding
    --backend vllm  one batched vLLM call per round, sampling from the model's generation config
"""
import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from sgl.segment.rules import palign_sentences

JUDGE_TEMPLATE = """
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


def judge_prompt(question: str, reasoning_part: str) -> str:
    return JUDGE_TEMPLATE.format(question=question, reasoning_part=reasoning_part)


def is_sufficient(response: str) -> bool:
    return "[ENOUGH]" in response or response.strip() == "ENOUGH"


@dataclass
class PrefixSearch:
    question: str
    sentences: list[str]
    left: int = 1
    right: int = 0
    best: int | None = None
    best_response: str = ""
    rounds: int = 0
    log: list = field(default_factory=list)

    def __post_init__(self):
        self.right = len(self.sentences)

    @property
    def done(self) -> bool:
        return self.left > self.right

    @property
    def mid(self) -> int:
        return (self.left + self.right) // 2

    def prompt(self) -> str:
        return judge_prompt(self.question, " ".join(self.sentences[: self.mid]))

    def update(self, response: str) -> None:
        mid, verdict = self.mid, is_sufficient(response)
        self.log.append((mid, verdict))
        self.rounds += 1
        if verdict:
            self.best, self.best_response, self.right = mid, response, mid - 1
        else:
            self.left = mid + 1

    def result(self) -> tuple[str, int, bool, str]:
        if self.best is None:
            return " ".join(self.sentences), len(self.sentences), False, ""
        return " ".join(self.sentences[: self.best]), self.best, True, self.best_response


def run_searches(searches: list[PrefixSearch], judge: Callable[[list[str]], list[str]],
                 batch_size: int = 512, on_round: Callable[[int, int], None] | None = None) -> None:
    """Advance every search to completion, asking `judge` about all open midpoints per round."""
    round_index = 0
    while True:
        active = [search for search in searches if not search.done]
        if not active:
            return
        round_index += 1
        if on_round:
            on_round(round_index, len(active))
        for start in range(0, len(active), batch_size):
            chunk = active[start:start + batch_size]
            responses = judge([search.prompt() for search in chunk])
            for search, response in zip(chunk, responses):
                search.update(response)


def apply_chat(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


class HFJudge:
    """Upstream chat(): one prompt per generate() call, the model's own generation defaults."""

    def __init__(self, model_name: str, max_new_tokens: int = 256):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto", device_map="auto",
                                                          trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_new_tokens = max_new_tokens

    def __call__(self, prompts: list[str]) -> list[str]:
        return [self.one(prompt) for prompt in prompts]

    def one(self, prompt: str) -> str:
        try:
            device = getattr(self.model, "device", None) or next(self.model.parameters()).device
            inputs = self.tokenizer([apply_chat(self.tokenizer, prompt)], return_tensors="pt").to(device)
            generated = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
            new_tokens = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
            return self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]
        except Exception as exc:  # upstream turns a model error into a NOT_ENOUGH verdict
            print(f"model error: {exc}")
            return f"ERROR: {exc}"


class VLLMJudge:
    def __init__(self, model_name: str, max_new_tokens: int = 256, gpu_memory_utilization: float = 0.8,
                 max_model_len: int | None = None):
        from transformers import AutoTokenizer
        from vllm import LLM

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.llm = LLM(model=model_name, trust_remote_code=True, gpu_memory_utilization=gpu_memory_utilization,
                       max_model_len=max_model_len)
        self.sampling = self.llm.get_default_sampling_params()  # = the model's generation_config
        self.sampling.max_tokens = max_new_tokens

    def __call__(self, prompts: list[str]) -> list[str]:
        texts = [apply_chat(self.tokenizer, prompt) for prompt in prompts]
        return [output.outputs[0].text for output in self.llm.generate(texts, self.sampling)]


def load_items(path: str, question_field: str, cot_field: str) -> list[tuple[int, dict]]:
    items = []
    for index, line in enumerate(open(path, encoding="utf-8")):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get(question_field) and row.get(cot_field):
            items.append((index, row))
    return items


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="the student, which judges sufficiency")
    parser.add_argument("--input", required=True, help="jsonl with the question and the teacher's long CoT")
    parser.add_argument("--output", required=True)
    parser.add_argument("--question-field", default="question")
    parser.add_argument("--cot-field", default="solution", help="upstream raw files call it Long-CoT")
    parser.add_argument("--answer-field", default="answer")
    parser.add_argument("--backend", choices=["hf", "vllm"], default="vllm")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=512, help="prompts per judge call (vllm)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    items = load_items(args.input, args.question_field, args.cot_field)[: args.limit]
    if args.backend == "hf":
        judge = HFJudge(args.model, args.max_new_tokens)
    else:
        judge = VLLMJudge(args.model, args.max_new_tokens, args.gpu_memory_utilization, args.max_model_len)

    searches = [PrefixSearch(row[args.question_field], palign_sentences(row[args.cot_field])) for _, row in items]
    run_searches(searches, judge, args.batch_size,
                 on_round=lambda r, n: print(f"round {r}: {n} open searches", flush=True))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for (index, row), search in zip(items, searches):
            prefix, length, ok, response = search.result()
            total = len(search.sentences)
            handle.write(json.dumps({
                "id": row.get("id", index),
                "answer": row.get(args.answer_field, ""),
                "question": search.question,
                "sufficient_reasoning": prefix,
                "sufficient_sentences": length,
                "total_sentences": total,
                "prefix_ratio": length / total if total else 0.0,
                "is_sufficient": ok,
                "evaluator_response": response,
            }, ensure_ascii=False) + "\n")
    ratios = [s.result()[1] / len(s.sentences) for s in searches if s.sentences]
    found = sum(s.best is not None for s in searches)
    print(f"{len(searches)} traces -> {output}; sufficient prefix found for {found}; "
          f"mean prefix ratio {sum(ratios) / max(len(ratios), 1):.3f}")


if __name__ == "__main__":
    main()
