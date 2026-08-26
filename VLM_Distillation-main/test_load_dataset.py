"""
Smoke-test for LazyVlmDistillDataset + VlmDistillDataCollator với real processors.

Usage
-----
# Student only:
python test_dataset.py \
    --data_path /data/train.jsonl \
    --image_dir  /data/images \
    --student_model Qwen/Qwen2-VL-2B-Instruct \
    --image_resolution mid \
    --max_len 512 \
    --batch_size 2

# Student + Teacher:
python test_dataset.py \
    --data_path /data/train.jsonl \
    --image_dir  /data/images \
    --student_model Qwen/Qwen2-VL-2B-Instruct \
    --teacher_model google/siglip2-base-patch16-224 \
    --image_resolution mid \
    --batch_size 2

# percent_data để chạy nhanh trên dataset lớn:
python test_dataset.py \
    --data_path /data/train.jsonl \
    --student_model Qwen/Qwen2-VL-2B-Instruct \
    --percent_data 0.01
"""

import argparse
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Optional

import torch

# ── project root vào sys.path (giống cách test model thật) ──────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── tắt log rác từ transformers/huggingface ──────────────────────────────────
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_VERBOSITY"] = "error"
for _name in ["httpcore", "httpx", "urllib3", "huggingface_hub", "transformers"]:
    _log = logging.getLogger(_name)
    _log.setLevel(logging.WARNING)
    _log.propagate = False

from src.arguments import DataArguments, ModelArguments
from src.model.processor import load_processor
from src.data.dataset import LazyVlmDistillDataset, VlmDistillDataCollator, _RESOLUTION_MAX_DIM

# ── ANSI colours ─────────────────────────────────────────────────────────────
GREEN  = "\033[32m"
YELLOW = "\033[33m"
RED    = "\033[31m"
CYAN   = "\033[36m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def ok(msg):      print(f"{GREEN}  ✓ {msg}{RESET}")
def warn(msg):    print(f"{YELLOW}  ⚠ {msg}{RESET}")
def fail(msg):    print(f"{RED}  ✗ {msg}{RESET}")
def section(msg): print(f"\n{BOLD}{CYAN}{'─'*60}\n  {msg}\n{'─'*60}{RESET}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Real-processor smoke-test for dataset.py")
    # data
    p.add_argument("--data_path",        required=True)
    p.add_argument("--image_dir",        default=None)
    p.add_argument("--image_resolution", default=None, choices=["high", "mid", "low"])
    p.add_argument("--max_len",          type=int,   default=None)
    p.add_argument("--percent_data",     type=float, default=1.0)
    # processors
    p.add_argument("--student_model",    required=True,
                   help="HuggingFace model name/path for student processor")
    p.add_argument("--teacher_model",    default=None,
                   help="HuggingFace model name/path for teacher processor (optional)")
    # collator
    p.add_argument("--batch_size",       type=int, default=2)
    return p.parse_args()


# ── helpers ───────────────────────────────────────────────────────────────────

def _tensor_summary(t: torch.Tensor) -> str:
    return f"shape={tuple(t.shape)}, dtype={t.dtype}, device={t.device}"


def _print_inputs(label: str, inputs: dict):
    print(f"    [{label}]")
    for k, v in inputs.items():
        if torch.is_tensor(v):
            print(f"      {k}: {_tensor_summary(v)}")
        else:
            print(f"      {k}: {type(v).__name__}")


# ── individual checks ─────────────────────────────────────────────────────────

def check_dataset_loads(data_args: DataArguments) -> LazyVlmDistillDataset:
    section("1 · Dataset loading")
    ds = LazyVlmDistillDataset(data_args=data_args)
    ok(f"Dataset created — {len(ds)} samples")
    if ds.skipped_missing_images:
        ok(f"Skipped {ds.skipped_missing_images} sample(s) with missing images")
    else:
        warn("skipped_missing_images = 0  (all image paths resolved or no images)")
    return ds


def check_lengths(ds: LazyVlmDistillDataset):
    section("2 · lengths / modality_lengths")
    lengths  = ds.lengths
    modality = ds.modality_lengths

    assert len(lengths)  == len(ds), "lengths size mismatch"
    assert len(modality) == len(ds), "modality_lengths size mismatch"
    assert all(l > 0 for l in lengths), "all lengths should be > 0"

    n_multimodal = sum(1 for v in modality if v > 0)
    n_text_only  = sum(1 for v in modality if v < 0)
    ok(f"lengths       — min={min(lengths)}, max={max(lengths)}, "
       f"mean={sum(lengths)/len(lengths):.1f}")
    ok(f"modality_lengths — {n_multimodal} multimodal, {n_text_only} text-only")


def check_getitem(ds: LazyVlmDistillDataset, n_samples: int = 3):
    section("3 · __getitem__ structure")
    for idx in range(min(n_samples, len(ds))):
        item = ds[idx]
        assert "id"         in item, "missing key 'id'"
        assert "messages"   in item, "missing key 'messages'"
        assert "image_path" in item, "missing key 'image_path'"

        msgs = item["messages"]
        assert msgs[0]["role"] == "system", "first message must be system prompt"

        has_image_block = any(
            any(b.get("type") == "image" for b in m.get("content", []))
            for m in msgs
        )
        is_multimodal = item["image_path"] is not None

        if is_multimodal:
            assert has_image_block, \
                f"[{idx}] has image_path but no image content block in messages"
            # verify PIL image object is actually there
            from PIL import Image
            for m in msgs:
                for b in m.get("content", []):
                    if b.get("type") == "image":
                        assert isinstance(b["image"], Image.Image), \
                            f"[{idx}] image block value is not a PIL.Image"
                        w, h = b["image"].size
                        ok(f"[{idx}] multimodal — {w}×{h} PIL image, "
                           f"{len(msgs)-1} turns")
                        break
        else:
            assert not has_image_block, \
                f"[{idx}] no image_path but image block found in messages"
            ok(f"[{idx}] text-only  — {len(msgs)-1} turns")


def check_image_resolution(ds: LazyVlmDistillDataset, resolution: Optional[str]):
    section("4 · Resolution cap")
    if resolution is None:
        warn("--image_resolution not set — skipping cap check")
        return

    max_dim = _RESOLUTION_MAX_DIM.get(resolution)
    if max_dim is None:
        warn(f"Unknown preset '{resolution}' — skipping cap check")
        return

    violations = 0
    checked    = 0
    for idx in range(len(ds)):
        item = ds[idx]
        if not item["image_path"]:
            continue
        for m in item["messages"]:
            for b in m.get("content", []):
                if b.get("type") == "image":
                    w, h = b["image"].size
                    checked += 1
                    if max(w, h) > max_dim:
                        fail(f"[{idx}] {w}×{h} exceeds max_dim={max_dim}")
                        violations += 1

    if violations:
        raise AssertionError(f"{violations} image(s) exceed the resolution cap")
    ok(f"All {checked} image(s) ≤ {max_dim}px on longest side ✓")


def check_collator(
    ds: LazyVlmDistillDataset,
    student_processor,
    teacher_processor,
    model_args: ModelArguments,
    batch_size: int,
):
    section("5 · Collator (real processors)")

    collator = VlmDistillDataCollator(
        student_processor=student_processor,
        teacher_processor=teacher_processor,
        data_args=ds.data_args,
        model_args=model_args,
    )

    indices   = list(range(min(batch_size, len(ds))))
    instances = [ds[i] for i in indices]
    batch     = collator(instances)

    # ── required keys ────────────────────────────────────────────────────────
    assert "ids"            in batch, "missing key 'ids'"
    assert "image_paths"    in batch, "missing key 'image_paths'"
    assert "student_inputs" in batch, "missing key 'student_inputs'"

    student_inp = batch["student_inputs"]
    assert "input_ids"      in student_inp, "student_inputs missing 'input_ids'"
    assert "attention_mask" in student_inp, "student_inputs missing 'attention_mask'"
    assert student_inp["input_ids"].dtype == torch.long, \
        f"input_ids dtype should be long, got {student_inp['input_ids'].dtype}"

    ok(f"Batch keys: {list(batch.keys())}")
    _print_inputs("student_inputs", student_inp)

    if collator.use_teacher:
        assert "teacher_inputs" in batch, "missing key 'teacher_inputs'"
        ok("teacher_inputs present ✓")
        _print_inputs("teacher_inputs", batch["teacher_inputs"])
    else:
        assert "teacher_inputs" not in batch, \
            "teacher_inputs should be absent when use_teacher=False"
        ok("teacher_inputs correctly absent ✓")

    # ── batch size sanity ────────────────────────────────────────────────────
    bs_actual = student_inp["input_ids"].shape[0]
    assert bs_actual == len(indices), \
        f"Expected batch_size={len(indices)}, got {bs_actual}"
    ok(f"Batch size = {bs_actual} ✓")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── DataArguments ────────────────────────────────────────────────────────
    data_args = DataArguments(
        data_path=args.data_path,
        image_dir=args.image_dir,
        image_resolution=args.image_resolution,
        max_len=args.max_len,
        percent_data=args.percent_data,
    )

    # ── ModelArguments & load processors ────────────────────────────────────
    section("0 · Loading processors")

    student_model_args = ModelArguments(
        model_name=args.student_model,
        teacher_model_name=args.teacher_model,  # None nếu không có teacher
    )

    print(f"  Loading student processor: {args.student_model}")
    student_processor = load_processor(student_model_args)
    ok(f"Student processor loaded — {type(student_processor).__name__}")

    teacher_processor = None
    if args.teacher_model:
        teacher_model_args = ModelArguments(model_name=args.teacher_model)
        print(f"  Loading teacher processor: {args.teacher_model}")
        teacher_processor = load_processor(teacher_model_args)
        ok(f"Teacher processor loaded — {type(teacher_processor).__name__}")
    else:
        warn("No --teacher_model given; teacher_inputs will not be tested")

    # ── run checks ───────────────────────────────────────────────────────────
    passed = failed = 0

    # check 1: dataset loading (phải pass mới chạy tiếp)
    try:
        ds = check_dataset_loads(data_args)
        passed += 1
    except Exception:
        fail("Dataset failed to load — aborting")
        traceback.print_exc()
        sys.exit(1)

    remaining = [
        ("lengths",          lambda: check_lengths(ds)),
        ("getitem",          lambda: check_getitem(ds)),
        ("image resolution", lambda: check_image_resolution(ds, args.image_resolution)),
        ("collator",         lambda: check_collator(
                                 ds,
                                 student_processor,
                                 teacher_processor,
                                 student_model_args,
                                 args.batch_size,
                             )),
    ]

    for name, fn in remaining:
        try:
            fn()
            passed += 1
        except Exception as exc:
            fail(f"Check '{name}' FAILED: {exc}")
            traceback.print_exc()
            failed += 1

    # ── summary ──────────────────────────────────────────────────────────────
    total = passed + failed
    section("Summary")
    status = f"{GREEN}{passed}/{total} passed{RESET}"
    if failed:
        status += f"  {RED}({failed} failed){RESET}"
    print(f"  {status}\n")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()