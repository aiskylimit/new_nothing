"""Segment-Selective SFT ports, checked against the untouched upstream code under references/."""
import json
import random
import types
from pathlib import Path

import pytest
import torch
from reference import SSFT, functions_from, load

from sgl.data.s1k import convert_rows, last_boxed
from sgl.segment.rules import split_segments
from sgl.ssft import attribution, build, select

ASSETS = Path(__file__).resolve().parent / "regression" / "assets"


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(ASSETS / "tokenizer"))


@pytest.fixture(scope="module")
def rows():
    rng = random.Random(0)
    out = []
    for line in (ASSETS / "palign_sample.jsonl").open():
        row = json.loads(line)
        solution = row["output"]
        segments = split_segments(solution, "paragraph")
        out.append({
            "question": row["input"] + "  ", "solution": solution, "segments": segments,
            "answer": last_boxed(solution) or "0",
            "selected_spans_ids": sorted(rng.sample(range(len(segments)), k=min(3, len(segments)))),
        })
    return out


# ------------------------------------------------------------------ data


def test_last_boxed_and_row_conversion_match_upstream():
    upstream = load(SSFT / "prepare_s1k.py", "upstream_prepare_s1k")
    for text in ["x \\boxed{1} y \\boxed{\\frac{1}{2}}", "\\boxed{unclosed", "none", "\\boxed{ a{b}c }"]:
        assert last_boxed(text) == upstream.last_boxed(text)
    source = [
        {"question": " q1 ", "deepseek_thinking_trajectory": "t \\boxed{5}", "solution": "5"},
        {"question": "q2", "deepseek_thinking_trajectory": "", "solution": "1"},
        {"question": "q3", "deepseek_thinking_trajectory": "no box", "solution": ""},
        {"question": "q4", "deepseek_thinking_trajectory": "\\boxed{" + "9" * 300 + "}", "solution": ""},
        {"question": "q5", "deepseek_thinking_trajectory": "\\boxed{2}", "solution": "3"},
    ]
    rows, stats = convert_rows(source, "deepseek_thinking_trajectory", "trace", 200)
    assert [r["question"] for r in rows] == ["q1", "q5"]
    assert [r["answer"] for r in rows] == ["5", "2"]
    assert stats == {"no_trace": 1, "no_answer": 1, "answer_too_long": 1, "trace_disagrees_with_gt": 1}
    rows, _ = convert_rows(source, "deepseek_thinking_trajectory", "gt", 200)
    assert [r["answer"] for r in rows] == ["5", "3"]


# ------------------------------------------------------------------ selection


def test_select_segments_matches_upstream_script(tmp_path, rows):
    """Run upstream get_important_segments.py and ours on the same synthetic IG rows."""
    import subprocess
    import sys

    rng = random.Random(1)
    data, ig = tmp_path / "segments.jsonl", tmp_path / "ig.jsonl"
    with data.open("w") as d, ig.open("w") as g:
        for index, row in enumerate(rows):
            d.write(json.dumps({k: row[k] for k in ("question", "solution", "answer", "segments")}) + "\n")
            compact = [[rng.randint(0, 9), round(rng.random(), 6), 0.0] for _ in row["segments"]]
            compact = [[n, a if n else 0.0, round(a * rng.uniform(-1, 1), 6) if n else 0.0] for n, a, _ in compact]
            if index == 0:  # no attribution signal: nothing selected
                compact = [[n, 0.0, 0.0] for n, _, _ in compact]
            if index == 3:  # upstream's full per-token format instead of the compact one
                g.write(json.dumps([[round(rng.uniform(-1, 1), 4) for _ in range(n)] for n, _, _ in compact]) + "\n")
            else:
                g.write(json.dumps({"segments": compact}) + "\n")
    expected = tmp_path / "upstream.jsonl"
    subprocess.run([sys.executable, str(SSFT / "Attribution" / "get_important_segments.py"),
                    "--input_data_file", str(data), "--IG_score_data_file", str(ig),
                    "--output_data_file", str(expected), "--cumulative_ratio", "0.6", "--coherence_max", "0.7"],
                   check=True, capture_output=True)
    ours = tmp_path / "ours.jsonl"
    select.main(["--data-path", str(data), "--ig", str(ig), "--output", str(ours),
                 "--cumulative-ratio", "0.6", "--coherence-max", "0.7"])
    assert ours.read_text() == expected.read_text()


# ------------------------------------------------------------------ build (train_mask.py labels)


def upstream_labels(tokenizer, rows, *, prompt_style, chat_kwargs, think, mask, max_len):
    namespace = {
        "args": types.SimpleNamespace(prompt_style=prompt_style, mask=mask, segment_mode="paragraph",
                                      max_seq_length=max_len),
        "tokenizer": tokenizer, "chat_kwargs": chat_kwargs, "think_str": think,
        "split_segments": split_segments, "SKIPPED": {"n": 0},
    }
    names = ["segment_char_bounds", "formatting_prompts_func"]
    functions_from(SSFT / "SelectiveSFT" / "train_mask.py", names, namespace)
    examples = {key: [row[key] for row in rows] for key in ("question", "solution", "selected_spans_ids")}
    out = namespace["formatting_prompts_func"](examples)
    return list(zip(out["input_ids"], out["labels"])), namespace["SKIPPED"]["n"]


@pytest.mark.parametrize("deepseek,think_prefix", [(False, "none"), (False, "plain"), (False, "off"),
                                                   (True, "none"), (True, "off")])
@pytest.mark.parametrize("prompt_style", ["default", "palign"])
@pytest.mark.parametrize("selective", [True, False])
@pytest.mark.parametrize("max_len", [32768, 700])
def test_build_matches_train_mask_labels(tokenizer, rows, deepseek, think_prefix, prompt_style, selective, max_len):
    chat_kwargs, think, _ = build.template_mode(deepseek, think_prefix)
    expected, skipped = upstream_labels(tokenizer, rows, prompt_style=prompt_style, chat_kwargs=chat_kwargs,
                                        think=think, mask=selective, max_len=max_len)
    ours = [build.build_record(tokenizer, row, prompt_style=prompt_style, chat_kwargs=chat_kwargs, think=think,
                               selective=selective, segment_mode="paragraph", max_seq_length=max_len)
            for row in rows]
    assert sum(record is None for record in ours) == skipped
    ours = [record for record in ours if record is not None]
    assert len(ours) == len(expected)
    for record, (input_ids, labels) in zip(ours, expected):
        assert record["input_ids"] == input_ids
        assert record["loss_mask"] == [int(label != -100) for label in labels]


def test_build_rejects_special_think_prefix_and_bad_templates(tokenizer):
    with pytest.raises(ValueError):
        build.template_mode(False, "special")
    chat_kwargs, think, expected = build.template_mode(True, "off")  # R1 ending, but a ChatML template
    with pytest.raises(SystemExit):
        build.check_template(tokenizer, chat_kwargs, think, expected)


# ------------------------------------------------------------------ attribution (Integrated Gradients)


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory, tokenizer):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    config = Qwen2Config(vocab_size=len(tokenizer), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=8192,
                         tie_word_embeddings=False, attention_dropout=0.0)
    path = tmp_path_factory.mktemp("tiny")
    Qwen2ForCausalLM(config).to(torch.bfloat16).save_pretrained(path)
    tokenizer.save_pretrained(path)
    return path


def test_build_example_spans_and_answer_range(tokenizer, rows):
    # Upstream locates the answer by scanning token strings for "boxed", "{" and "}", which assumes
    # "{" and the closing "}" are separate tokens (true for Qwen's tokenizer). Kept as is for
    # fidelity; a plain numeric answer keeps this check tokenizer independent.
    row = dict(rows[1], answer="42")
    ids, spans, (start, end) = attribution.build_example(tokenizer, row)
    nonempty = [span for span in spans if span != (0, 0)]
    assert tokenizer.decode(ids[nonempty[0][0]:nonempty[-1][1]]) == "".join(row["segments"])
    assert all(a < b for a, b in nonempty) and all(b == c for (_, b), (c, _) in zip(nonempty, nonempty[1:]))
    assert tokenizer.decode(ids[start:end]) == "42"


def test_integrated_gradients_match_upstream(monkeypatch, tiny_model_dir, rows):
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *a, **k: self)  # upstream hard-codes .cuda()
    upstream = load(SSFT / "Attribution" / "grad_analyze.py", "upstream_grad_analyze")
    reference = upstream.IntegratedGradientsAttribution(str(tiny_model_dir), gradient_checkpointing=True)
    model, tokenizer = attribution.load_model(str(tiny_model_dir), "bfloat16", None, True)
    ours = attribution.IntegratedGradients(model, steps=7, batch_size=3)
    for row in rows[:3]:
        row = dict(row, segments=row["segments"][:12], answer="42")
        ids, spans, answer = attribution.build_example(tokenizer, row)
        expected = reference.batch_compute_step_to_answer_attribution_integrated(
            ids, spans, answer, baseline_token_id=tokenizer.pad_token_id, steps=7, batch_size=3)
        got = ours(ids, spans, answer, tokenizer.pad_token_id)
        assert got == expected
        assert attribution.compact(got)[0][0] == spans[0][1] - spans[0][0]


def test_attribution_prompt_ids_are_the_rendered_chat_template(tokenizer, rows):
    """Upstream ran transformers 4.x, where apply_chat_template(tokenize=True) returns the ids of the
    rendered template without extra special tokens; transformers 5 returns a BatchEncoding instead."""
    row = dict(rows[0], answer="7")
    ids, spans, _ = attribution.build_example(tokenizer, row)
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": attribution.USER_TEMPLATE.format(input=row["question"])}],
        tokenize=False, add_generation_prompt=True,
    )
    prompt_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    assert ids[:len(prompt_ids)] == prompt_ids
    assert spans[0][0] == len(prompt_ids)
