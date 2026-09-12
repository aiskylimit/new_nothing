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
MAX_REFERENCE_RESPONSE_TOKENS = 2048


def truncate_reference_response(response, tokenizer):
    """Limit every reference before prompt rendering, keeping a Unicode-safe prefix."""
    if len(tokenizer.encode(response, add_special_tokens=False)) <= MAX_REFERENCE_RESPONSE_TOKENS:
        return response
    low, high = 1, len(response) - 1
    fitted = None
    while low <= high:
        middle = (low + high) // 2
        candidate = response[:middle]
        if len(tokenizer.encode(candidate, add_special_tokens=False)) <= MAX_REFERENCE_RESPONSE_TOKENS:
            if candidate.strip():
                fitted = candidate
            low = middle + 1
        else:
            high = middle - 1
    if fitted is None:
        raise ValueError("Reference response token budget cannot fit a nonempty reference response")
    return fitted


def context_generation_prompt(record, tokenizer, template, max_prompt_length=None):
    """Cap every reference first, then fit the complete prompt to its token budget."""
    if max_prompt_length is not None and max_prompt_length < 1:
        raise ValueError("--max-prompt-length must be positive")
    prompt = get_raw_prompt(record, tokenizer)
    response = get_response(record)
    if isinstance(response, list):
        response = response[0] if response else None
    if not isinstance(prompt, str) or not prompt.strip() or not isinstance(response, str) or not response.strip():
        raise ValueError("Context preparation requires a nonempty user_prompt/instruction/prompt "
                         "and original output/response/generated_text")
    if record.get("system_prompt"):
        prompt = record["system_prompt"] + "\n\n" + prompt
    response = truncate_reference_response(response, tokenizer)

    def encode(reference):
        text = template.format(prompt=prompt, response=reference)
        if getattr(tokenizer, "chat_template", None):
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": text}], tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
        return tokenizer.encode(text, add_special_tokens=False)

    ids = encode(response)
    if max_prompt_length is None or len(ids) <= max_prompt_length:
        return ids

    base_ids = encode("")
    if len(base_ids) > max_prompt_length:
        # A long question can overflow even after removing the reference. Use
        # the same prompt-suffix policy as training, retaining the generation
        # header at the end instead of rejecting the dataset row.
        return ids[-max_prompt_length:]

    # Search character prefixes to avoid decoding partial Unicode tokens. Always
    # measure the complete rendered prompt, including BPE boundary effects and
    # any repeated {response} placeholders in a custom template. Keep only a
    # verified fit: token counts need not be strictly monotonic across prefixes.
    low, high = 1, len(response) - 1
    fitted_ids = None
    while low <= high:
        middle = (low + high) // 2
        reference = response[:middle]
        candidate = encode(reference)
        if len(candidate) <= max_prompt_length:
            if reference.strip():
                fitted_ids = candidate
            low = middle + 1
        else:
            high = middle - 1
    if fitted_ids is None:
        return ids[-max_prompt_length:]
    return fitted_ids


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
                        help="Context-generation input budget; reference responses are always capped "
                             "at 2048 teacher tokens first, then shortened further; if still too long, "
                             "keep the rendered prompt suffix within this budget")
    parser.add_argument("--max-length", type=int, default=4096,
                        help="Teacher context-generation total sequence budget")
    # Accept old command lines; training now applies its own truncation budgets.
    parser.add_argument("--t-max-prompt-length", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--student-max-length", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--t-max-length", type=int, default=None, help=argparse.SUPPRESS)
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
    ids = context_generation_prompt(record, tokenizer, args.context_generation_template,
                                    max_prompt_length=args.max_prompt_length)
    if not ids:
        raise ValueError("Context-generation prompt must contain at least one token")
    return ids[-args.max_prompt_length:]


def preflight_dataset(args, tokenizer):
    """Scan all rows before GPU generation; retain neither tokens nor records in RAM."""
    source, _ = validate_args(args)
    count = 0
    for count, record in enumerate(read_jsonl(source), 1):
        try:
            checked_generation_prompt(args, record, tokenizer)
            text = render_privileged_teacher_prompt(record, tokenizer, "", args.privileged_context_template)
            if not tokenizer.encode(text, add_special_tokens=False):
                raise ValueError("Training teacher prompt must contain at least one token")
        except (ValueError, TypeError, KeyError) as error:
            raise ValueError(f"{source}: row {count}: {error}") from error
    if count == 0:
        raise ValueError(f"Training dataset is empty: {source}")
    return count


def validate_model_lengths(args, config):
    limit = getattr(config, "max_position_embeddings", None)
    if isinstance(limit, int) and limit > 0 and args.max_length > limit:
        raise ValueError(f"--max-length exceeds teacher max_position_embeddings={limit}")


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
                # validate_args reserves max_new_tokens; checked prompts are
                # already bounded so overlong rows cannot exceed max_length.
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
                    prepared = dict(record)
                    prepared[args.privileged_context_field] = context
                    # Validate the complete training prompt without imposing a
                    # token budget on privileged distillation.
                    try:
                        build_privileged_teacher_input(
                            prepared, tokenizer, args.privileged_context_field,
                            args.privileged_context_template)
                    except ValueError as error:
                        raise ValueError(f"{source}: row {count + offset}: {error}") from error
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
