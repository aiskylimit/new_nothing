import argparse
import json
import math
import os
from pathlib import Path
import tempfile

import torch
from tqdm import tqdm

from data_utils.prepared_privileged import fingerprint, read_jsonl, validate_context_field
from data_utils.records import get_raw_prompt, get_response
from data_utils.privileged import (DEFAULT_CONTEXT_TEMPLATE, build_privileged_teacher_input,
                                   render_privileged_teacher_prompt)


DEFAULT_GENERATION_TEMPLATE = (
    "Create concise supporting context for the following question, using the reference response "
    "to identify relevant facts, definitions, useful reasoning hints, formulas or instruction guidelines. "
    "Return only the supporting context, not a replacement answer or a full worked solution.\n\n"
    "Question:\n{prompt}\n\nReference response:\n{response}\n\nSupporting context:"
)


def context_generation_prompt(record, tokenizer, template):
    prompt = get_raw_prompt(record, tokenizer)
    response = get_response(record)
    if isinstance(response, list):
        response = response[0] if response else None
    if not isinstance(prompt, str) or not prompt.strip() or not isinstance(response, str) or not response.strip():
        raise ValueError("Context preparation requires a nonempty user_prompt/instruction/prompt "
                         "and original output/response/generated_text")
    if record.get("system_prompt"):
        prompt = record["system_prompt"] + "\n\n" + prompt
    text = template.format(prompt=prompt, response=response)
    if getattr(tokenizer, "chat_template", None):
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
    return tokenizer.encode(text, add_special_tokens=False)


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True,
                        help="Full raw dataset JSONL (before splitting), or a directory containing train.jsonl")
    parser.add_argument("--output", required=True, help="New prepared JSONL file, separate from the source")
    parser.add_argument("--teacher-model-path", required=True)
    parser.add_argument("--teacher-peft-path", default=None)
    parser.add_argument("--device-map", default="auto", help="Hugging Face device map; auto can shard the teacher across GPUs")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-prompt-length", type=int, default=3072,
                        help="Context-generation input budget, including question, reference response and instruction")
    parser.add_argument("--max-length", type=int, default=4096,
                        help="Teacher context-generation total sequence budget")
    parser.add_argument("--t-max-prompt-length", type=int, default=1536,
                        help="Training teacher prompt budget after context insertion; match finetune_v2")
    parser.add_argument("--student-max-length", type=int, default=1024,
                        help="Match finetune_v2 --max-length; conservatively reserve this much response space")
    parser.add_argument("--t-max-length", type=int, default=None,
                        help="Training teacher total budget; defaults to t-max-prompt-length + student-max-length")
    parser.add_argument("--privileged-context-field", default="context",
                        help="JSONL field stored alongside the original prompt and response")
    parser.add_argument("--privileged-context-template", default=DEFAULT_CONTEXT_TEMPLATE,
                        help="Match the context insertion template used by finetune_v2")
    parser.add_argument("--context-generation-template", default=DEFAULT_GENERATION_TEMPLATE,
                        help="Teacher instruction containing {prompt} and {response}")
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.)
    parser.add_argument("--top-p", type=float, default=1.)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def validate_args(args):
    if args.batch_size < 1 or args.max_new_tokens < 1:
        raise ValueError("--batch-size and --max-new-tokens must be positive")
    if not 0 < args.max_prompt_length < args.max_length:
        raise ValueError("Require 0 < --max-prompt-length < --max-length")
    if args.max_prompt_length + args.max_new_tokens > args.max_length:
        raise ValueError("Require --max-prompt-length + --max-new-tokens <= --max-length")
    if args.t_max_prompt_length < 1 or args.student_max_length < 2:
        raise ValueError("Require positive --t-max-prompt-length and --student-max-length >= 2")
    if args.t_max_length is None:
        args.t_max_length = args.t_max_prompt_length + args.student_max_length
    if args.t_max_length < args.t_max_prompt_length + args.student_max_length:
        raise ValueError("Reserve the full student response: --t-max-length must be >= "
                         "--t-max-prompt-length + --student-max-length")
    if "{privileged_context}" not in args.privileged_context_template:
        raise ValueError("--privileged-context-template must contain {privileged_context}")
    validate_context_field(args.privileged_context_field)
    if "{prompt}" not in args.context_generation_template or "{response}" not in args.context_generation_template:
        raise ValueError("--context-generation-template must contain {prompt} and {response}")
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        raise ValueError("--temperature must be finite and positive")
    if not math.isfinite(args.top_p) or not 0 < args.top_p <= 1 or args.top_k < 0:
        raise ValueError("Require 0 < --top-p <= 1 and --top-k >= 0")
    source = Path(args.data_dir)
    source = source / "train.jsonl" if source.is_dir() else source
    output = Path(args.output)
    if source.resolve() == output.resolve():
        raise ValueError("Prepared output must be separate from the canonical training source")
    if output.exists():
        raise FileExistsError(f"Prepared output already exists: {output}; choose a new output path")
    if not source.is_file():
        raise FileNotFoundError(f"Prepare requires the full source JSONL: {source}")
    return source, output


def checked_generation_prompt(args, record, tokenizer):
    ids = context_generation_prompt(record, tokenizer, args.context_generation_template)
    if not ids or len(ids) > args.max_prompt_length:
        raise ValueError(f"Context-generation prompt has {len(ids)} tokens, limit is "
                         f"{args.max_prompt_length}; increase --max-prompt-length and --max-length "
                         "to retain the full question/reference")
    return ids


def preflight_dataset(args, tokenizer):
    """Scan all rows before GPU generation; retain neither tokens nor records in RAM."""
    source, _ = validate_args(args)
    count = 0
    for count, record in enumerate(read_jsonl(source), 1):
        try:
            checked_generation_prompt(args, record, tokenizer)
            text = render_privileged_teacher_prompt(record, tokenizer, "", args.privileged_context_template)
            size = len(tokenizer.encode(text, add_special_tokens=False))
            if not size or size > args.t_max_prompt_length:
                raise ValueError(f"Training teacher prompt already has {size} tokens without context, "
                                 f"limit is {args.t_max_prompt_length}; increase --t-max-prompt-length "
                                 "and --t-max-length in both prepare and training")
        except (ValueError, TypeError, KeyError) as error:
            raise ValueError(f"{source}: row {count}: {error}") from error
    if count == 0:
        raise ValueError(f"Training dataset is empty: {source}")
    return count


def validate_model_lengths(args, config):
    limit = getattr(config, "max_position_embeddings", None)
    if isinstance(limit, int) and limit > 0:
        for name in ("max_length", "t_max_length"):
            if getattr(args, name) > limit:
                raise ValueError(f"--{name.replace('_', '-')} exceeds teacher max_position_embeddings={limit}")


@torch.no_grad()
def prepare_full_dataset(args, teacher, tokenizer, *, preflight_done=False):
    source, output = validate_args(args)
    if not preflight_done:
        preflight_dataset(args, tokenizer)
    validate_model_lengths(args, getattr(teacher, "config", None))
    if tokenizer.eos_token_id is None:
        raise ValueError("Teacher tokenizer must define eos_token_id")
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    device = teacher.get_input_embeddings().weight.device
    output.parent.mkdir(parents=True, exist_ok=True)
    was_training = teacher.training
    teacher.eval()
    count = 0
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent,
                                         prefix=output.name + ".", suffix=".partial", delete=False) as handle:
            temporary = Path(handle.name)

            def generate_rows(records):
                prompts = [checked_generation_prompt(args, row, tokenizer) for row in records]
                width = max(map(len, prompts))
                if width + args.max_new_tokens > args.max_length:
                    raise ValueError("Prepare prompt + --max-new-tokens exceeds --max-length")
                ids = torch.full((len(prompts), width), pad_id, dtype=torch.long, device=device)
                mask = torch.zeros_like(ids)
                for index, prompt in enumerate(prompts):
                    ids[index, -len(prompt):] = torch.tensor(prompt, device=device)
                    mask[index, -len(prompt):] = 1
                options = dict(max_new_tokens=args.max_new_tokens, do_sample=args.do_sample,
                               pad_token_id=pad_id, eos_token_id=tokenizer.eos_token_id,
                               return_dict_in_generate=True, output_scores=False,
                               num_return_sequences=1, num_beams=1, use_cache=True)
                if args.do_sample:
                    options.update(temperature=args.temperature, top_p=args.top_p, top_k=args.top_k)
                sequences = teacher.generate(input_ids=ids, attention_mask=mask, **options).sequences
                if sequences.shape[0] != len(records):
                    raise ValueError("Teacher must return exactly one context per source row")
                for offset, (record, generated) in enumerate(zip(records, sequences[:, width:]), 1):
                    context_ids = generated.tolist()
                    if tokenizer.eos_token_id in context_ids:
                        context_ids = context_ids[:context_ids.index(tokenizer.eos_token_id) + 1]
                    context = tokenizer.decode(context_ids, skip_special_tokens=True).strip()
                    if not context:
                        raise ValueError("Teacher returned empty privileged context; adjust generation settings and rerun prepare")
                    prepared = dict(record)
                    prepared[args.privileged_context_field] = context
                    # Retokenize the COMPLETE training prompt: BPE at insertion
                    # boundaries means isolated context length is not sufficient.
                    try:
                        build_privileged_teacher_input(
                            prepared, tokenizer, args.privileged_context_field,
                            args.privileged_context_template, args.t_max_prompt_length)
                    except ValueError as error:
                        raise ValueError(f"{source}: row {count + offset}: {error}; "
                                         "increase teacher training limits in both stages "
                                         "or reduce --max-new-tokens") from error
                    prepared["privileged_preparation"] = dict(
                        version=1, kind="context", source_sha256=fingerprint(record),
                        teacher_model_path=args.teacher_model_path,
                        teacher_peft_path=args.teacher_peft_path,
                        prompt_format="raw_question_v2",
                    )
                    handle.write(json.dumps(prepared, ensure_ascii=False) + "\n")

            batch = []
            for record in tqdm(read_jsonl(source), desc="Preparing full dataset context", unit="rows"):
                batch.append(record)
                if len(batch) == args.batch_size:
                    generate_rows(batch)
                    count += len(batch)
                    batch = []
            if batch:
                generate_rows(batch)
                count += len(batch)
            if not count:
                raise ValueError("Training dataset is empty")
            handle.flush()
            os.fsync(handle.fileno())
        # Publish only after full generation completes; refuse concurrent overwrite.
        os.link(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        teacher.train(was_training)
    print(f"Prepared {count} / {count} dataset rows -> {output}")
    return count


def main():
    args = get_parser().parse_args()
    validate_args(args)
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise ValueError("Launch prepare with python, not torchrun; use --device-map auto for multiple GPUs")
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, set_seed
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model_path)
    config = AutoConfig.from_pretrained(args.teacher_model_path)
    validate_model_lengths(args, config)
    print(f"Validated {preflight_dataset(args, tokenizer)} rows before loading teacher weights")
    dtype = args.dtype if args.dtype == "auto" else getattr(torch, args.dtype)
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher_model_path, torch_dtype=dtype, device_map=args.device_map)
    if args.teacher_peft_path:
        from peft import PeftModel
        teacher = PeftModel.from_pretrained(teacher, args.teacher_peft_path).merge_and_unload()
    teacher.requires_grad_(False)
    prepare_full_dataset(args, teacher, tokenizer, preflight_done=True)


if __name__ == "__main__":
    main()
