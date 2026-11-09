"""Đánh giá Pass@1 trên GSM8K / MATH500 / AIME24 / AIME25 (spec §8).

Mặc định dùng vLLM (nhanh, hỗ trợ LoRA) cho mọi phương pháp để bảng chính so
sánh trên **cùng một engine và cùng sampling config**. Ours ở mode ``dynamic``
phải chạy bằng HF loop nên có engine riêng — dùng cho subset/ablation.

    # baseline
    python -m eval.run_eval --method vanilla_sft --adapter outputs/vanilla_sft
    # ours (merged adapter, vLLM)
    python -m eval.run_eval --method ours --ours_mode merged \\
        --phase1_dir outputs/phase1 --gate_dir outputs/ours
    # ours (dynamic routing, HF, subset)
    python -m eval.run_eval --method ours_dynamic --ours_mode dynamic --engine hf \\
        --benchmarks gsm8k math500 --limit 200
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Optional

import numpy as np

from core.data_utils import track_cli_provided, write_json
from core.proctitle import set_proc_title
from core.model_utils import load_tokenizer
from core.local_assets import local_adapter_path, local_model_path
from core.segmentation import render_prompt
from eval.benchmarks import BENCHMARKS, EvalItem, load_benchmark
from eval.grade import pass_at_1, pass_at_k

DEFAULT_BENCHMARKS = ["math500", "aime24", "aime25", "amc"]

# P-ALIGN (arXiv 2601.10064) gắn chỉ dẫn này vào mọi câu hỏi trước chat template.
PALIGN_ANSWER_INSTRUCTION = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pass@1 evaluation")
    parser.add_argument("--config", default=None)
    parser.add_argument("--method", required=True, help="Tên hàng trong bảng kết quả")
    parser.add_argument("--model_name", required=True, help="Bắt buộc, không có model mặc định (tránh eval nhầm base model)")
    parser.add_argument("--adapter", default=None, help="Thư mục LoRA adapter (baseline)")
    parser.add_argument("--benchmarks", nargs="+", default=DEFAULT_BENCHMARKS)
    parser.add_argument("--engine", default="vllm", choices=["vllm", "hf"])
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--max_model_len", type=int, default=None,
                        help="Override max_model_len cua vLLM (mac dinh = max_tokens+2048). "
                             "Dung de tai hien dung setting P-ALIGN (max_model_len=max_tokens, "
                             "khong cong buffer prompt).")
    parser.add_argument("--repetition_penalty", type=float, default=1.05,
                        help="Theo P-ALIGN scripts/Inference.sh (1.05)")
    parser.add_argument("--answer_instruction", default=PALIGN_ANSWER_INSTRUCTION,
                        help="Chỉ dẫn gắn trước mỗi câu hỏi; đặt '' để bỏ")
    parser.add_argument("--disable_thinking", action="store_true",
                        help="Truyen enable_thinking=False vao chat template (chi anh huong "
                             "eval, khong dung cho train) -- thu nghiem cho model hybrid-"
                             "thinking nhu Qwen3 khi muon tra loi thang, khong sinh <think>.")
    parser.add_argument("--k", type=int, default=3,
                        help="Số sample mỗi bài (P-ALIGN: 3, dùng cho cả Pass@1 và Pass@3)")
    parser.add_argument("--limit", type=int, default=None, help="Số bài mỗi benchmark")
    parser.add_argument("--curiosft_data_dir", default="data/curiosft_raw",
                        help="Thư mục parquet cho benchmark cs_* (kịch bản CurioSFT)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--save_generations", action="store_true")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--max_lora_rank", type=int, default=16)
    # Ours
    parser.add_argument("--ours_mode", default=None,
                        choices=["merged", "per_example", "dynamic"])
    parser.add_argument("--phase1_dir", default="outputs/phase1")
    parser.add_argument("--gate_dir", default="outputs/ours")
    parser.add_argument("--M", type=int, default=3)
    parser.add_argument("--lora_r", type=int, default=16,
                        help="Rank của từng expert — dùng để suy ra rank sau khi gộp")
    parser.add_argument("--combination_type", default="cat", choices=["cat", "linear", "svd"],
                        help="'cat' tái tạo chính xác sum_m g_m dW_m; xem eval/gate_inference.py")
    parser.add_argument("--merge_dtype", default="float32", choices=["float32", "bfloat16"],
                        help="dtype lúc gộp adapter; bfloat16 nếu thiếu RAM (~28GB)")
    parser.add_argument("--merged_dir", default=None)
    return track_cli_provided(parser)


VLLM_LORA_RANKS = [8, 16, 32, 64, 128, 256]


def _round_up_lora_rank(rank: int) -> int:
    """vLLM chỉ chấp nhận một số mức max_lora_rank rời rạc."""
    for allowed in VLLM_LORA_RANKS:
        if rank <= allowed:
            return allowed
    raise ValueError(f"rank {rank} vượt mức vLLM hỗ trợ ({VLLM_LORA_RANKS[-1]})")


def resolve_k(name: str, override: Optional[int]) -> int:
    if override:
        return override
    return BENCHMARKS.get(name, {}).get("k", 1)
    
from concurrent.futures import ProcessPoolExecutor 

def _grade_task(args_tuple):
    comps, answer, kind = args_tuple
    return pass_at_1(comps, answer, kind=kind), pass_at_k(comps, answer, kind=kind)


def _describe_tokens(lengths: List[int]) -> dict:
    """Tính toán min, max, mean, median, std cho danh sách độ dài token."""
    if not lengths:
        return {"count": 0, "mean": 0.0, "min": 0, "max": 0, "median": 0.0, "std": 0.0}
    arr = np.asarray(lengths, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": round(float(arr.mean()), 2),
        "std": round(float(arr.std()), 2),
        "min": int(arr.min()),
        "p25": round(float(np.percentile(arr, 25)), 2),
        "median": round(float(np.median(arr)), 2),
        "p75": round(float(np.percentile(arr, 75)), 2),
        "max": int(arr.max()),
    }


def main() -> None:
    args = parse_args()
    args.model_name = local_model_path(args.model_name)
    set_proc_title("eval-" + args.method)
    tokenizer = load_tokenizer(args.model_name)

    adapter_path = args.adapter
    if args.ours_mode == "merged":
        from eval.gate_inference import export_merged_adapter

        merged_dir = args.merged_dir or os.path.join(args.gate_dir, "merged_adapter")
        adapter_path = export_merged_adapter(
            args.model_name, args.phase1_dir, args.gate_dir, merged_dir, args.M,
            combination_type=args.combination_type, merge_dtype=args.merge_dtype,
        )
        # combination_type="cat" nâng rank lên M*r; vLLM chỉ nhận vài mức rank
        # rời rạc nên làm tròn LÊN mức hợp lệ gần nhất.
        if args.combination_type == "cat":
            needed = args.lora_r * args.M
            args.max_lora_rank = _round_up_lora_rank(max(args.max_lora_rank, needed))
            print(f"[eval] merged rank = {needed} -> max_lora_rank = {args.max_lora_rank}")

    if args.engine == "hf" and args.ours_mode in {"per_example", "dynamic"}:
        runner = _HFGateRunner(args)
    elif args.engine == "vllm":
        runner = _VLLMRunner(args, adapter_path)
    else:
        raise ValueError(
            "engine=hf chỉ hỗ trợ Ours (per_example/dynamic). "
            "Baseline dùng --engine vllm."
        )

    started = time.time()
    per_benchmark: Dict[str, dict] = {}
    n_workers = min(32, os.cpu_count() or 1)
    all_tok_lens: List[int] = []
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for name in args.benchmarks:
            items = load_benchmark(name, limit=args.limit,
                                   curiosft_data_dir=args.curiosft_data_dir)
            k = resolve_k(name, args.k)
            instr = (args.answer_instruction + " ") if args.answer_instruction else ""
            questions = [instr + i.question for i in items]
            enable_thinking = False if args.disable_thinking else None
            prompts = [render_prompt(tokenizer, q, enable_thinking=enable_thinking) for q in questions]
            completions = runner.generate(prompts, k=k, questions=questions)

            tasks = [(comps, item.answer, getattr(item, "kind", "math"))
                     for item, comps in zip(items, completions)]

            # chunksize=4 giúp cân bằng tải tốt cho cả benchmark 30 bài lẫn 5000 bài
            graded = list(executor.map(_grade_task, tasks, chunksize=4))
            scores = [g[0] for g in graded]
            scores_passk = [g[1] for g in graded]

            # Tính thống kê token của output (completions)
            b_tok_lens: List[int] = []
            problem_tok_lens: List[List[int]] = []
            for comps in completions:
                p_lens = [len(tokenizer(c, add_special_tokens=False)["input_ids"]) for c in comps]
                b_tok_lens.extend(p_lens)
                problem_tok_lens.append(p_lens)
            all_tok_lens.extend(b_tok_lens)
            token_stats = _describe_tokens(b_tok_lens)
    
            accuracy = sum(scores) / len(scores) if scores else 0.0
            pass_k = sum(scores_passk) / len(scores_passk) if scores_passk else 0.0
            per_benchmark[name] = {
                "pass@1": accuracy,
                f"pass@{k}": pass_k,
                "n_problems": len(items),
                "k": k,
                "token_stats": token_stats,
                "per_problem": scores,
                "per_problem_passk": scores_passk,
            }
            print(f"[eval] {args.method} | {name}: Pass@1 = {accuracy*100:.2f} "
                  f"| Pass@{k} = {pass_k*100:.2f} "
                  f"({len(items)} bài, k={k}) | "
                  f"Tokens: mean={token_stats['mean']:.1f} (min={token_stats['min']}, max={token_stats['max']})")
    
            if args.save_generations:
                _dump_generations(args, name, items, completions, scores, problem_tok_lens)

    out_path = os.path.join(args.results_dir, f"{args.method}.json")

   
    merged_benchmarks: Dict[str, dict] = {}
    if os.path.exists(out_path):
        try:
            with open(out_path, encoding="utf-8") as handle:
                merged_benchmarks = json.load(handle).get("benchmarks", {}) or {}
        except (json.JSONDecodeError, OSError):
            merged_benchmarks = {}
    merged_benchmarks.update(per_benchmark)

    scores = [v["pass@1"] for v in merged_benchmarks.values()]
    passk_scores = [v[key] for v in merged_benchmarks.values()
                    for key in v if key.startswith("pass@") and key != "pass@1"]
    overall_tokens = _describe_tokens(all_tok_lens)
    result = {
        "method": args.method,
        "model_name": args.model_name,
        "adapter": adapter_path,
        "engine": args.engine,
        "ours_mode": args.ours_mode,
        "sampling": {"temperature": args.temperature, "top_p": args.top_p,
                     "max_tokens": args.max_tokens, "seed": args.seed,
                     "repetition_penalty": args.repetition_penalty,
                     "answer_instruction": args.answer_instruction},
        "benchmarks": merged_benchmarks,
        "avg": sum(scores) / len(scores) if scores else 0.0,
        "avg_passk": sum(passk_scores) / len(passk_scores) if passk_scores else 0.0,
        "overall_tokens": overall_tokens,
        "runtime_sec": time.time() - started,
    }
    write_json(out_path, result)
    print(f"[eval] Avg = {result['avg']*100:.2f} | "
          f"Overall Output Tokens: mean={overall_tokens['mean']:.1f} (min={overall_tokens['min']}, max={overall_tokens['max']}) -> {out_path}")


# --------------------------------------------------------------------------- #
# Engines
# --------------------------------------------------------------------------- #
class _VLLMRunner:
    """vLLM + LoRA. Một engine cho mọi phương pháp -> so sánh công bằng."""

    def __init__(self, args, adapter_path: Optional[str]):
        from vllm import LLM

        self.args = args
        self.adapter_path = local_adapter_path(adapter_path) if adapter_path else None
        self.llm = LLM(
            model=local_model_path(args.model_name),
            dtype="bfloat16",
            enable_lora=adapter_path is not None,
            max_lora_rank=args.max_lora_rank,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len if args.max_model_len is not None else args.max_tokens + 2048,
            seed=args.seed,
            enable_prefix_caching=True,
            max_num_seqs=256,
            max_num_batched_tokens=16384,
        )

    def generate(self, prompts: List[str], k: int, questions=None) -> List[List[str]]:
        from vllm import SamplingParams
        from vllm.lora.request import LoRARequest

        params = SamplingParams(
            n=k,
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            repetition_penalty=getattr(self.args, "repetition_penalty", 1.0),
            max_tokens=self.args.max_tokens,
            seed=self.args.seed,
            stop=["<|im_end|>", "<|endoftext|>", "</s>"],
        )
        lora_request = (
            LoRARequest("adapter", 1, self.adapter_path) if self.adapter_path else None
        )
        outputs = self.llm.generate(prompts, params, lora_request=lora_request)
        return [[o.text for o in out.outputs] for out in outputs]


class _HFGateRunner:
    """HF loop có step-level routing (Ours mode per_example/dynamic)."""

    def __init__(self, args):
        from eval.gate_inference import GateRouter, SamplingConfig

        self.args = args
        self.router = GateRouter(
            args.model_name, args.phase1_dir, args.gate_dir, n_experts=args.M
        )
        self.sampling = SamplingConfig(
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_tokens,
            seed=args.seed,
        )

    def generate(self, prompts: List[str], k: int, questions=None) -> List[List[str]]:
        from eval.gate_inference import SamplingConfig

        results: List[List[str]] = []
        for index, question in enumerate(questions or prompts):
            samples = []
            for sample_index in range(k):
                config = SamplingConfig(
                    temperature=self.sampling.temperature,
                    top_p=self.sampling.top_p,
                    max_new_tokens=self.sampling.max_new_tokens,
                    seed=self.sampling.seed + 1000 * sample_index,
                )
                samples.append(
                    self.router.generate(question, config, mode=self.args.ours_mode)
                )
            results.append(samples)
            if (index + 1) % 10 == 0:
                print(f"[eval/hf] {index + 1}/{len(prompts)}")
        return results


def _dump_generations(args, name: str, items: List[EvalItem], completions, scores, token_lengths: Optional[List[List[int]]] = None) -> None:
    path = os.path.join(args.results_dir, "generations", f"{args.method}_{name}.jsonl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for i, (item, comps, score) in enumerate(zip(items, completions, scores)):
            row = {
                "index": item.index, "question": item.question,
                "gold": item.answer, "completions": comps, "pass@1": score,
            }
            if token_lengths is not None and i < len(token_lengths):
                row["token_counts"] = token_lengths[i]
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
