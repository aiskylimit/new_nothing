import argparse
import json
import os
import pickle
from datetime import datetime
from pathlib import Path
import tempfile
import multiprocessing

from tqdm import tqdm

from data_utils.prepared_privileged import fingerprint, read_jsonl, validate_context_field
from data_utils.records import get_raw_prompt, get_response
from data_utils.privileged import (DEFAULT_CONTEXT_TEMPLATE, build_privileged_teacher_input,
                                   render_privileged_teacher_prompt)
from vllm import SamplingParams


DEFAULT_GENERATION_TEMPLATE = (
    "Create concise supporting context for the following question, using the reference response "
    "to identify relevant facts, definitions, useful reasoning hints, formulas or instruction guidelines. "
    "Return only the supporting context, not a replacement answer or a full worked solution.\n\n"
    "Question:\n{prompt}\n\nReference response:\n{response}\n\nSupporting context:"
)
DEFAULT_MAX_REFERENCE_RESPONSE_TOKENS = 2048
GENERATION_BACKUP_SUFFIX_GLOB = ".generation_backup.*.pkl"


def _binary_search_max_prefix(text, fits_fn):
    """Tìm prefix dài nhất của `text` sao cho fits_fn(prefix) == True.

    Lưu ý: giả định số token không giảm khi prefix dài thêm. Với hầu hết
    tokenizer BPE điều này đúng, nhưng byte-fallback ở biên grapheme có thể
    vi phạm giả định này ở một vài điểm hiếm. Thuật toán vẫn an toàn (không
    crash) nhưng có thể không tìm ra prefix hợp lệ dài nhất tuyệt đối.
    """
    low, high = 1, len(text) - 1
    fitted = None
    while low <= high:
        middle = (low + high) // 2
        candidate = text[:middle]
        if candidate.strip() and fits_fn(candidate):
            fitted = candidate
            low = middle + 1
        else:
            high = middle - 1
    return fitted


def truncate_reference_response(response, tokenizer, max_tokens):
    if len(tokenizer.encode(response, add_special_tokens=False)) <= max_tokens:
        return response

    # Thu hẹp không gian tìm kiếm trước để binary search chạy nhanh trên chuỗi dài
    upper_bound_chars = max_tokens * 10
    if len(response) > upper_bound_chars:
        response = response[:upper_bound_chars]

    fitted = _binary_search_max_prefix(
        response,
        lambda candidate: len(tokenizer.encode(candidate, add_special_tokens=False)) <= max_tokens,
    )
    if fitted is None:
        raise ValueError("Reference response token budget cannot fit a nonempty reference response")
    return fitted


def context_generation_prompt(record, tokenizer, template, max_reference_response_tokens, max_prompt_length=None):
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
    response = truncate_reference_response(response, tokenizer, max_reference_response_tokens)

    def encode(prompt_text, reference):
        text = template.format(prompt=prompt_text, response=reference)
        if getattr(tokenizer, "chat_template", None):
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": text}], tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
        return tokenizer.encode(text, add_special_tokens=False)

    ids = encode(prompt, response)
    if max_prompt_length is None or len(ids) <= max_prompt_length:
        return ids

    base_ids = encode(prompt, "")
    if len(base_ids) > max_prompt_length:
        # Câu hỏi (+ system_prompt) + khung template một mình đã vượt ngân sách:
        # không còn chỗ cho reference response. Thay vì raise lỗi và làm hỏng cả
        # batch, cắt bớt chính câu hỏi cho vừa ngân sách — vẫn tốt hơn bỏ hẳn dòng
        # dữ liệu này, nhưng in cảnh báo rõ để biết dòng nào bị mất thông tin.
        fitted_prompt = _binary_search_max_prefix(
            prompt,
            lambda candidate: len(encode(candidate, "")) <= max_prompt_length,
        )
        if fitted_prompt is None:
            # Ngay cả template rỗng + 1 ký tự câu hỏi cũng không vừa -> template
            # tự nó đã vượt ngân sách, không thể cứu được bằng cách cắt prompt.
            raise ValueError(
                "--context-generation-template alone exceeds --max-prompt-length "
                f"({len(encode('', '')) } > {max_prompt_length}); "
                "increase --max-prompt-length or shorten the template."
            )
        print(
            f"[canh bao] cau hoi bi cat bot vi prompt+template vuot --max-prompt-length "
            f"({len(base_ids)} > {max_prompt_length} tokens); "
            f"con lai {len(fitted_prompt)}/{len(prompt)} ky tu cua cau hoi goc.",
            flush=True,
        )
        prompt = fitted_prompt
        return encode(prompt, "")

    fitted = _binary_search_max_prefix(
        response,
        lambda candidate: len(encode(prompt, candidate)) <= max_prompt_length,
    )
    if fitted is None:
        # Không tìm được prefix non-empty nào vừa; dùng response rỗng (đã biết vừa)
        return base_ids
    return encode(prompt, fitted)


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True,
                        help="Full raw dataset JSONL (before splitting), or a directory containing train.jsonl")
    parser.add_argument("--output", required=True, help="New prepared JSONL file, separate from the source")
    parser.add_argument("--teacher-model-path", required=True)
    parser.add_argument("--teacher-peft-path", default=None)
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                        help="vLLM tensor-parallel degree; shards the teacher across this many GPUs")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9,
                        help="Fraction of GPU memory vLLM is allowed to reserve for weights + KV cache")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-prompt-length", type=int, default=3072,
                        help="Context-generation input budget")
    parser.add_argument("--max-length", type=int, default=4096,
                        help="Teacher context-generation total sequence budget")
    parser.add_argument("--max-reference-response-tokens", type=int, default=DEFAULT_MAX_REFERENCE_RESPONSE_TOKENS,
                        help="Hard cap on reference-response tokens before it's fed into the generation prompt")
    parser.add_argument("--privileged-context-field", default="context",
                        help="JSONL field stored alongside the original prompt and response")
    parser.add_argument("--privileged-context-template", default=DEFAULT_CONTEXT_TEMPLATE,
                        help="Match the context insertion template used by finetune_v2")
    parser.add_argument("--context-generation-template", default=DEFAULT_GENERATION_TEMPLATE,
                        help="Teacher instruction containing {prompt} and {response}")
    parser.add_argument("--num-workers", type=int, default=min(multiprocessing.cpu_count(), 64),
                        help="CPU worker processes for tokenization/validation pass")
    parser.add_argument("--chunksize", type=int, default=100,
                        help="imap chunksize for the CPU preparation pool")
    parser.add_argument("--ignore-generation-backup", action="store_true",
                        help="Bo qua moi generation backup co san va luon chay lai Giai doan 2 tren GPU")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def validate_args(args):
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be positive")
    if args.tensor_parallel_size < 1:
        raise ValueError("--tensor-parallel-size must be positive")
    if not 0 < args.gpu_memory_utilization <= 1:
        raise ValueError("--gpu-memory-utilization must be in (0, 1]")
    if not 0 < args.max_prompt_length < args.max_length:
        raise ValueError("Require 0 < --max-prompt-length < --max-length")
    if args.max_prompt_length + args.max_new_tokens > args.max_length:
        raise ValueError("Require --max-prompt-length + --max-new-tokens <= --max-length")
    if args.max_reference_response_tokens < 1:
        raise ValueError("--max-reference-response-tokens must be positive")
    if args.num_workers < 1:
        raise ValueError("--num-workers must be positive")
    if args.chunksize < 1:
        raise ValueError("--chunksize must be positive")
    if "{privileged_context}" not in args.privileged_context_template:
        raise ValueError("--privileged-context-template must contain {privileged_context}")
    validate_context_field(args.privileged_context_field)
    if "{prompt}" not in args.context_generation_template or "{response}" not in args.context_generation_template:
        raise ValueError("--context-generation-template must contain {prompt} and {response}")
    source = Path(args.data_dir)
    source = source / "train.jsonl" if source.is_dir() else source
    output = Path(args.output)
    if source.resolve() == output.resolve():
        raise ValueError("Prepared output must be separate from the canonical training source")
    if output.exists() and not output.is_file():
        raise ValueError(f"--output path exists and is not a regular file: {output}")
    if not source.is_file():
        raise FileNotFoundError(f"Prepare requires the full source JSONL: {source}")
    return source, output


# ---------------------------------------------------------
# MULTIPROCESSING WORKER CONFIGURATION
# ---------------------------------------------------------
_worker_tokenizer = None
_worker_args = None


def _init_worker(model_path, args):
    """Khởi tạo tokenizer riêng cho mỗi worker process."""
    global _worker_tokenizer, _worker_args
    from transformers import AutoTokenizer
    _worker_tokenizer = AutoTokenizer.from_pretrained(model_path)
    _worker_args = args


def _process_single_row(record_tuple):
    idx, record = record_tuple
    try:
        prompt_ids = context_generation_prompt(
            record, _worker_tokenizer, _worker_args.context_generation_template,
            max_reference_response_tokens=_worker_args.max_reference_response_tokens,
            max_prompt_length=_worker_args.max_prompt_length,
        )
        if not prompt_ids:
            return False, idx, None, "Context-generation prompt must contain at least one token"

        # Preflight validation: đảm bảo prompt huấn luyện (với context rỗng placeholder)
        # cũng mã hóa được, để lỗi lộ ra sớm ở giai đoạn CPU thay vì sau khi tốn GPU generate.
        text = render_privileged_teacher_prompt(record, _worker_tokenizer, "", _worker_args.privileged_context_template)
        if not _worker_tokenizer.encode(text, add_special_tokens=False):
            return False, idx, None, "Training teacher prompt must contain at least one token"

        return True, idx, {"record": record, "prompt_token_ids": prompt_ids}, None
    except Exception as error:
        return False, idx, None, str(error)


def validate_model_lengths(args, config):
    limit = getattr(config, "max_position_embeddings", None)
    if isinstance(limit, int) and limit > 0 and args.max_length > limit:
        raise ValueError(f"--max-length exceeds teacher max_position_embeddings={limit}")


def build_sampling_params(args):
    return SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=0.0,
    )


def build_lora_request(args):
    if not args.teacher_peft_path:
        return None
    from vllm.lora.request import LoRARequest
    return LoRARequest("teacher_adapter", 1, args.teacher_peft_path)


def prepare_records(source, args):
    """Giai đoạn 1: đọc + tokenize + validate toàn bộ dataset bằng CPU multiprocessing.

    Chạy trên B200 nên toàn bộ dataset được nạp hết vào RAM một lần —
    đơn giản hơn streaming và không phải nút thắt trên các server này.
    """
    print("Giai đoạn 1: đọc và mã hóa toàn bộ dữ liệu (CPU multiprocessing)...")
    raw_records = list(enumerate(read_jsonl(source), 1))
    if not raw_records:
        raise ValueError(f"Training dataset is empty: {source}")

    all_records = []
    all_prompts = []
    with multiprocessing.Pool(processes=args.num_workers, initializer=_init_worker,
                              initargs=(args.teacher_model_path, args)) as pool:
        for success, idx, result, err_msg in tqdm(
            pool.imap(_process_single_row, raw_records, chunksize=args.chunksize),
            total=len(raw_records), desc="Preparing data",
        ):
            if not success:
                raise ValueError(f"{source}: row {idx}: {err_msg}")
            all_records.append(result["record"])
            all_prompts.append({"prompt_token_ids": result["prompt_token_ids"]})

    print(f"Đã chuẩn bị {len(all_prompts)} prompts.")
    return all_records, all_prompts


def run_teacher_generation(args, all_prompts):
    """Giai đoạn 2: nạp teacher model bằng vLLM và generate toàn bộ batch."""
    from vllm import LLM

    teacher = LLM(
        model=args.teacher_model_path,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_length,
        enable_lora=bool(args.teacher_peft_path),
        seed=args.seed,
    )

    sampling_params = build_sampling_params(args)
    lora_request = build_lora_request(args)

    print("Giai đoạn 2: teacher generation (vLLM continuous batching)...")
    outputs = teacher.generate(
        all_prompts,
        sampling_params=sampling_params,
        lora_request=lora_request,
        use_tqdm=True,
    )
    return outputs


def extract_generated_texts(source, outputs):
    """Chuyển output vLLM (RequestOutput) thành list[str] gọn nhẹ để backup/ghi file.

    Tách riêng bước này để backup không phải pickle nguyên object vLLM (nặng,
    không đảm bảo ổn định giữa các version) mà chỉ lưu text thuần.
    """
    texts = []
    for offset, generated in enumerate(outputs, 1):
        if len(generated.outputs) != 1:
            raise ValueError(f"{source}: row {offset}: teacher must return exactly one completion")
        texts.append(generated.outputs[0].text.strip())
    return texts


def find_latest_generation_backup(output):
    candidates = sorted(output.parent.glob(output.name + GENERATION_BACKUP_SUFFIX_GLOB))
    return candidates[-1] if candidates else None


def load_generation_backup(output):
    """Nếu đã có sẵn file backup từ lần chạy trước, load lại generated_texts.

    Trả về None nếu không tìm thấy backup nào (sẽ phải generate lại từ đầu).
    """
    backup_path = find_latest_generation_backup(output)
    if backup_path is None:
        return None
    print(f"Phat hien generation backup co san: {backup_path}")
    with open(backup_path, "rb") as f:
        payload = pickle.load(f)
    texts = payload.get("generated_texts")
    if not isinstance(texts, list):
        raise ValueError(f"{backup_path}: backup khong hop le (thieu 'generated_texts'), xoa file nay roi chay lai.")
    return texts


def save_generation_backup(output, generated_texts, args):
    """Backup raw teacher outputs (chỉ text) ra đĩa để có thể resume nếu Giai đoạn 3 lỗi."""
    output.parent.mkdir(parents=True, exist_ok=True)
    backup_path = output.parent / f"{output.name}.generation_backup.{datetime.now():%Y%m%d_%H%M%S}.pkl"
    try:
        with open(backup_path, "wb") as f:
            pickle.dump(
                {
                    "generated_texts": generated_texts,
                    "num_outputs": len(generated_texts),
                    "args": vars(args),
                },
                f,
            )
        print(f"Da backup raw outputs -> {backup_path}")
    except Exception as error:
        print(f"Canh bao: backup outputs that bai ({error}), tiep tuc ghi file chinh...")


def write_prepared_dataset(source, output, args, tokenizer, all_records, generated_texts):
    """Giai đoạn 3: validate + ghi file JSONL atomically (temp file + hardlink)."""
    if len(generated_texts) != len(all_records):
        raise ValueError(
            f"So luong context da generate ({len(generated_texts)}) khong khop so records "
            f"({len(all_records)}); neu dang resume tu backup, backup co the khong khop voi "
            f"dataset hien tai."
        )

    print("Giai đoạn 3: tổng hợp và ghi file...")
    output.parent.mkdir(parents=True, exist_ok=True)

    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent,
                                         prefix=output.name + ".", suffix=".partial", delete=False) as handle:
            temporary = Path(handle.name)

            for offset, (record, context) in tqdm(enumerate(zip(all_records, generated_texts), 1),
                                                    total=len(all_records), desc="Writing file"):
                context = context.strip() if isinstance(context, str) else ""
                prepared = dict(record)
                if not context:
                    context = "<empty>"
                prepared[args.privileged_context_field] = context

                try:
                    build_privileged_teacher_input(
                        prepared, tokenizer, args.privileged_context_field,
                        args.privileged_context_template)
                except ValueError as error:
                    raise ValueError(f"{source}: row {offset}: {error}") from error

                prepared["privileged_preparation"] = dict(
                    version=1,
                    kind="context",
                    source_sha256=fingerprint(record),
                    teacher_model_path=args.teacher_model_path,
                    teacher_peft_path=args.teacher_peft_path,
                    prompt_format="raw_question_v2",
                )
                handle.write(json.dumps(prepared, ensure_ascii=False) + "\n")

            handle.flush()
            os.fsync(handle.fileno())

        if output.exists():
            print(f"Canh bao: --output da ton tai, se bi ghi de: {output}")

        os.replace(temporary, output)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

    print(f"Hoàn thành: đã ghi {len(all_records)} rows -> {output}")


def main():
    args = get_parser().parse_args()
    source, output = validate_args(args)
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise ValueError("Launch prepare with python, not torchrun; use --tensor-parallel-size for multiple GPUs")

    from transformers import AutoConfig, AutoTokenizer

    config = AutoConfig.from_pretrained(args.teacher_model_path)
    validate_model_lengths(args, config)

    all_records, all_prompts = prepare_records(source, args)

    generated_texts = None if args.ignore_generation_backup else load_generation_backup(output)
    if generated_texts is not None:
        print("Da tim thay generation backup -> bo qua Giai doan 2 (khong chay lai teacher tren GPU).")
    else:
        outputs = run_teacher_generation(args, all_prompts)
        generated_texts = extract_generated_texts(source, outputs)
        save_generation_backup(output, generated_texts, args)

    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model_path)
    write_prepared_dataset(source, output, args, tokenizer, all_records, generated_texts)


if __name__ == "__main__":
    main()