"""Step segmentation and step -> token-span alignment.

Notation (theo spec §3.2):
    N   = số step trong một mẫu
    s_i = step thứ i
    T_i = số token của step i

Mọi span token trong file này là nửa mở ``[tok_start, tok_end)`` và nằm trong
toạ độ của **full input_ids** (prompt + response), không phải toạ độ riêng của
phần thinking. Đây là phần dễ bug nhất của repo nên có assert ở
``validate_step_spans``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Iterable, List, Optional, Sequence, Tuple

PARAGRAPH_SPLIT = "\n\n"

# Dùng cho segmentation mode = "sentence" (ablation §9).
_SENTENCE_END = re.compile(r"(?<=[.!?])[\s]+|\n+")


@dataclass
class StepSpan:
    """Một step: vị trí ký tự trong thinking text + vị trí token trong full seq."""

    index: int
    char_start: int
    char_end: int
    tok_start: int = -1
    tok_end: int = -1

    @property
    def n_tokens(self) -> int:
        return self.tok_end - self.tok_start

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# 1. Text -> step (char spans)
# --------------------------------------------------------------------------- #
def split_thinking_char_spans(text: str, mode: str = "paragraph") -> List[Tuple[int, int]]:
    """Tách ``text`` thành các step, trả về char span nửa mở.

    ``mode="paragraph"`` (mặc định, đúng spec §3.2): split theo ``"\\n\\n"``.
    ``mode="sentence"``: split theo ranh giới câu — dùng cho ablation §9.

    Khoảng trắng giữa các step bị bỏ (không thuộc step nào); span trả về đã
    strip hai đầu và không rỗng.
    """
    if mode == "paragraph":
        raw_spans = _split_by_separator(text, PARAGRAPH_SPLIT)
    elif mode == "sentence":
        raw_spans = _split_by_regex(text, _SENTENCE_END)
    else:
        raise ValueError(f"unknown segmentation mode: {mode!r}")

    spans: List[Tuple[int, int]] = []
    for start, end in raw_spans:
        start, end = _strip_span(text, start, end)
        if end > start:
            spans.append((start, end))
    return spans


def _split_by_separator(text: str, sep: str) -> List[Tuple[int, int]]:
    spans, cursor = [], 0
    while True:
        hit = text.find(sep, cursor)
        if hit < 0:
            spans.append((cursor, len(text)))
            return spans
        spans.append((cursor, hit))
        cursor = hit + len(sep)


def _split_by_regex(text: str, pattern: re.Pattern) -> List[Tuple[int, int]]:
    spans, cursor = [], 0
    for match in pattern.finditer(text):
        spans.append((cursor, match.start()))
        cursor = match.end()
    spans.append((cursor, len(text)))
    return spans


def _strip_span(text: str, start: int, end: int) -> Tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def split_thinking_into_steps(text: str, mode: str = "paragraph") -> List[str]:
    """Bản tiện lợi trả về list string (dùng trong ``data/prepare.py``)."""
    return [text[a:b] for a, b in split_thinking_char_spans(text, mode)]


# --------------------------------------------------------------------------- #
# 2. Char spans -> token spans
# --------------------------------------------------------------------------- #
def map_char_spans_to_token_spans(
    offsets: Sequence[Tuple[int, int]],
    char_spans: Sequence[Tuple[int, int]],
    token_index_offset: int = 0,
) -> List[Tuple[int, int]]:
    """Ánh xạ char span -> token span dựa trên ``offsets`` của tokenizer.

    ``offsets[k] = (char_start, char_end)`` của token thứ k **trong text đã
    tokenize** (thường là phần thinking). ``token_index_offset`` được cộng vào
    kết quả để đưa về toạ độ full sequence.

    Mỗi token được gán cho step có phần giao ký tự lớn nhất; token rơi vào
    khoảng trắng giữa hai step (giao = 0 với mọi step) bị bỏ. Token đặc biệt có
    ``offset == (0, 0)`` cũng bị bỏ.
    """
    assignment: List[int] = [-1] * len(offsets)
    for k, (t_start, t_end) in enumerate(offsets):
        if t_end <= t_start:  # special token / empty offset
            continue
        best_step, best_overlap = -1, 0
        for i, (c_start, c_end) in enumerate(char_spans):
            overlap = min(t_end, c_end) - max(t_start, c_start)
            if overlap > best_overlap:
                best_step, best_overlap = i, overlap
        assignment[k] = best_step

    spans: List[Tuple[int, int]] = []
    for i in range(len(char_spans)):
        members = [k for k, a in enumerate(assignment) if a == i]
        if not members:
            # Step quá ngắn để chiếm trọn một token (hiếm) -> span rỗng, caller lọc.
            spans.append((token_index_offset, token_index_offset))
            continue
        spans.append((token_index_offset + members[0], token_index_offset + members[-1] + 1))
    return spans


def validate_step_spans(
    spans: Sequence[Tuple[int, int]],
    region_start: int,
    region_end: int,
    strict_cover: bool = False,
) -> None:
    """Assert §14: span không chồng nhau, tăng dần, nằm trong vùng thinking.

    ``strict_cover=True`` bắt buộc hợp của các span phủ kín ``[region_start,
    region_end)`` — chỉ đúng khi tokenizer không sinh token nằm hoàn toàn trong
    khoảng trắng phân tách; mặc định ta chỉ cảnh báo qua giá trị trả về của
    :func:`coverage_report`.
    """
    prev_end = region_start
    for i, (start, end) in enumerate(spans):
        if end < start:
            raise AssertionError(f"step {i}: end < start ({start}, {end})")
        if start < region_start or end > region_end:
            raise AssertionError(
                f"step {i}: span ({start}, {end}) ngoài vùng thinking "
                f"[{region_start}, {region_end})"
            )
        if start < prev_end:
            raise AssertionError(f"step {i}: span ({start}, {end}) chồng lên step trước")
        prev_end = max(prev_end, end)
    if strict_cover:
        covered = sum(end - start for start, end in spans)
        if covered != region_end - region_start:
            raise AssertionError(
                f"span phủ {covered} token, vùng thinking có {region_end - region_start}"
            )


def coverage_report(
    spans: Sequence[Tuple[int, int]], region_start: int, region_end: int
) -> dict:
    """Thống kê độ phủ, dùng để log/sanity-check thay vì assert cứng."""
    total = max(region_end - region_start, 0)
    covered = sum(max(end - start, 0) for start, end in spans)
    return {
        "n_steps": len(spans),
        "region_tokens": total,
        "covered_tokens": covered,
        "uncovered_tokens": total - covered,
        "coverage": (covered / total) if total else 0.0,
    }


# --------------------------------------------------------------------------- #
# 3. Tokenize một mẫu train và trả về step span trong full sequence
# --------------------------------------------------------------------------- #
@dataclass
class SegmentedExample:
    """Một mẫu train đã tokenize kèm ranh giới step."""

    input_ids: List[int]
    labels: List[int]              # -100 ở phần prompt
    step_spans: List[Tuple[int, int]]   # chỉ phần thinking
    think_span: Tuple[int, int]         # vùng thinking trong full sequence
    answer_span: Tuple[int, int]        # phần sau </think> (weight = 1)
    prompt_len: int
    coverage: dict

    @property
    def n_steps(self) -> int:
        return len(self.step_spans)


def build_segmented_example(
    tokenizer,
    question: str,
    thinking: str,
    answer: str,
    *,
    segmentation_mode: str = "paragraph",
    cutoff_len: int = 4096,
    min_step_tokens: int = 1,
) -> SegmentedExample:
    """Ghép prompt + response, tokenize, và ánh xạ step -> token span.

    Response có dạng ``<think>\\n{thinking}\\n</think>\\n\\n{answer}`` (định dạng
    của DeepSeek-R1-Distill). Loss được tính trên toàn bộ response; phần sau
    ``</think>`` là một segment riêng luôn có trọng số 1.
    """
    prompt_text = render_prompt(tokenizer, question)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

    # Một số chat template của R1-Distill đã tự nối "<think>" vào generation
    # prompt. Không nối thêm lần nữa.
    think_open = "" if prompt_text.rstrip().endswith("<think>") else "<think>\n"
    think_close = "\n</think>\n\n"
    open_ids = tokenizer(think_open, add_special_tokens=False)["input_ids"] if think_open else []
    close_ids = tokenizer(think_close, add_special_tokens=False)["input_ids"]

    think_enc = tokenizer(thinking, add_special_tokens=False, return_offsets_mapping=True)
    think_ids = think_enc["input_ids"]
    think_offsets = list(think_enc["offset_mapping"])

    answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
    eos_id = tokenizer.eos_token_id
    if eos_id is not None:
        answer_ids = answer_ids + [eos_id]

    input_ids = prompt_ids + open_ids + think_ids + close_ids + answer_ids
    input_ids = input_ids[:cutoff_len]

    think_start = len(prompt_ids) + len(open_ids)
    think_end = think_start + len(think_ids)
    answer_start = think_end
    answer_end = answer_start + len(close_ids) + len(answer_ids)

    # Cắt theo cutoff: mọi biên phải nằm trong độ dài thực tế.
    total = len(input_ids)
    think_end = min(think_end, total)
    answer_start = min(answer_start, total)
    answer_end = min(answer_end, total)

    char_spans = split_thinking_char_spans(thinking, segmentation_mode)
    tok_spans = map_char_spans_to_token_spans(think_offsets, char_spans, think_start)
    tok_spans = [
        (start, min(end, think_end))
        for start, end in tok_spans
        if start < think_end and min(end, think_end) - start >= min_step_tokens
    ]

    validate_step_spans(tok_spans, think_start, think_end)
    coverage = coverage_report(tok_spans, think_start, think_end)

    labels = [-100] * len(prompt_ids) + input_ids[len(prompt_ids) :]
    labels = labels[:total]

    return SegmentedExample(
        input_ids=input_ids,
        labels=labels,
        step_spans=tok_spans,
        think_span=(think_start, think_end),
        answer_span=(answer_start, answer_end),
        prompt_len=len(prompt_ids),
        coverage=coverage,
    )


def render_prompt(tokenizer, question: str, enable_thinking: Optional[bool] = None) -> str:
    """Prompt theo chat template của model; fallback nếu template thiếu.

    Dùng chung cho train (`build_segmented_example`) và eval (`eval/run_eval.py`)
    để prompt lúc train và lúc infer giống hệt nhau.

    ``enable_thinking``: None (mặc định) = không truyền tham số này, giữ nguyên
    hành vi cũ (không ảnh hưởng train). Chỉ set True/False khi cần thử nghiệm
    riêng cho eval (vd Qwen3 hybrid-thinking) — model không hỗ trợ tham số này
    sẽ tự bỏ qua nhờ except bên dưới.
    """
    messages = [{"role": "user", "content": question}]
    kwargs = {}
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **kwargs
        )
    except Exception:  # tokenizer không có chat template (hoặc không hỗ trợ enable_thinking)
        if kwargs:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return f"User: {question}\nAssistant:"


def spans_to_lengths(spans: Iterable[Tuple[int, int]]) -> List[int]:
    return [end - start for start, end in spans]
