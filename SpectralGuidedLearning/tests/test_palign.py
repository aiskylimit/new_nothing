"""P-ALIGN ports, checked against the untouched upstream code under references/ and its released data."""
import gzip
import json
import random
import types
from pathlib import Path

import pytest
from reference import PALIGN, ROOT, load, stub_module

from sgl.eval.graders import palign as palign_grader
from sgl.palign import align, build, truncate

ASSETS = Path(__file__).resolve().parent / "regression" / "assets"
RELEASED = ROOT / "P-ALIGN" / "data" / "palign_sft_qwen2.5-7b.json.gz"


@pytest.fixture(scope="module")
def upstream_search():
    return load(PALIGN / "binary_search.py", "upstream_binary_search")


def test_judge_prompt_is_upstreams(upstream_search, monkeypatch):
    seen = []
    monkeypatch.setattr(upstream_search, "chat", lambda prompt, model: seen.append(prompt) or "[ENOUGH]")
    response, ok = upstream_search.reasoning_sufficiency_check("What is 1+1?", "One plus one. It is two.")
    assert seen == [truncate.judge_prompt("What is 1+1?", "One plus one. It is two.")]
    assert ok and truncate.is_sufficient(response)


@pytest.mark.parametrize("response,verdict", [
    ("[ENOUGH]", True), ("  ENOUGH \n", True), ("[NOT_ENOUGH]", False), ("NOT ENOUGH", False),
    ("ENOUGH.", False), ("I think [ENOUGH] here", True), ("ERROR: boom", False),
])
def test_verdict_rule(upstream_search, monkeypatch, response, verdict):
    monkeypatch.setattr(upstream_search, "chat", lambda prompt, model: response)
    assert upstream_search.reasoning_sufficiency_check("q", "r")[1] == verdict == truncate.is_sufficient(response)


def test_lockstep_search_finds_what_the_sequential_search_finds(upstream_search, monkeypatch):
    """Any monotone or non-monotone judge: each trace sees the same sequence of verdicts."""
    rng = random.Random(0)
    cases = []
    for index in range(300):
        n = rng.choice([0, 1, 2, 3, 7, 16, 33, 100])
        sentences = [f"s{index}-{k}." for k in range(n)]
        kind = rng.choice(["threshold", "never", "noisy"])
        threshold = rng.randint(1, max(n, 1))
        seed = rng.random()
        cases.append((f"q{index}", sentences, kind, threshold, seed))

    def oracle(case, prefix):
        _, sentences, kind, threshold, seed = case
        k = len(prefix.split(" ")) if prefix else 0
        if kind == "never":
            return "[NOT_ENOUGH]"
        if kind == "noisy":
            return "[ENOUGH]" if random.Random(f"{seed}-{k}").random() < 0.5 else "nope"
        return "[ENOUGH]" if k >= threshold else "[NOT_ENOUGH]"

    by_question = {case[0]: case for case in cases}
    monkeypatch.setattr(upstream_search, "reasoning_sufficiency_check",
                        lambda q, part: (lambda r: (r, truncate.is_sufficient(r)))(oracle(by_question[q], part)))
    monkeypatch.setattr(upstream_search, "print", lambda *a, **k: None, raising=False)
    expected = [upstream_search.find_minimal_sufficient_prefix(q, s, sleep_sec=0) for q, s, *_ in cases]

    searches = [truncate.PrefixSearch(q, s) for q, s, *_ in cases]

    def judge(prompts):
        out = []
        for prompt in prompts:
            question = prompt.split("Question:\n", 1)[1].split("\n", 1)[0]
            part = prompt.split("Partial reasoning:\n", 1)[1][:-1]
            out.append(oracle(by_question[question], part))
        return out

    truncate.run_searches(searches, judge, batch_size=37)
    assert [search.result() for search in searches] == expected


def test_continuation_prompt_is_upstreams(tmp_path):
    captured = []

    class FakeLLM:
        def generate(self, texts, sampling):
            captured.extend(texts)
            return [types.SimpleNamespace(outputs=[types.SimpleNamespace(text="cont")]) for _ in texts]

    class FakeTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return messages[0]["content"]

    class _Ctx:
        def __init__(self, rows):
            self.rows = rows

        def __enter__(self):
            return self.rows

        def __exit__(self, *args):
            return False

    upstream = load(PALIGN / "prefix-alignment.py", "upstream_prefix_alignment", stubs={
        "vllm": stub_module("vllm", LLM=object, SamplingParams=object),
        "jsonlines": stub_module("jsonlines", open=lambda path: _Ctx([json.loads(line) for line in open(path)])),
    })
    source = tmp_path / "truncated.jsonl"
    items = [{"question": "Q one?", "sufficient_reasoning": "Step. Step two."},
             {"question": "Q {two}", "sufficient_reasoning": "x \\boxed{1}"}]
    source.write_text("".join(json.dumps(item) + "\n" for item in items))
    upstream.process_data(str(source), str(tmp_path / "out.jsonl"), FakeLLM(), 8, FakeTokenizer(), None)
    assert captured == [align.continue_prompt(i["question"], i["sufficient_reasoning"]) for i in items]


def test_grader_reproduces_palign_published_labels():
    rows = [json.loads(line) for line in (ASSETS / "palign_labels.jsonl").open()]
    assert len(rows) == 60
    for row in rows:
        assert palign_grader.grade_math_verify(row["outputs"], row["answer"]) == row["labels"]


def test_assemble_reproduces_the_released_marker_layout():
    released = json.load(gzip.open(RELEASED))
    assert len(released) == 966
    for row in released:
        head, continuation = row["output"].split(build.END + "\n", 1)
        assert head.startswith(build.BEGIN)
        prefix = head[len(build.BEGIN):]
        assert build.assemble(prefix, continuation) == row["output"]
        assert row["instruction"] == build.INSTRUCTION


def test_build_filters_by_answer(tmp_path):
    truncated = tmp_path / "t.jsonl"
    aligned = tmp_path / "a.jsonl"
    truncated.write_text("".join(json.dumps(r) + "\n" for r in [
        {"question": "q1", "answer": "4"}, {"question": "q2", "answer": "5"}, {"question": "q3", "answer": ""}]))
    aligned.write_text("".join(json.dumps(r) + "\n" for r in [
        {"question": "q1", "sufficient_reasoning": "Two plus two.", "output": " so \\boxed{4}\n"},
        {"question": "q2", "sufficient_reasoning": "x.", "output": "\\boxed{6}"},
        {"question": "q3", "sufficient_reasoning": "y.", "output": "\\boxed{1}"}]))
    out = tmp_path / "data.json"
    build.main(["--aligned", str(aligned), "--truncated", str(truncated), "--output", str(out)])
    rows = json.loads(out.read_text())
    assert [r["input"] for r in rows] == ["q1"]
    assert rows[0]["output"] == "<Begin_of_Prefix>Two plus two.<End_of_Prefix>\nso \\boxed{4}"
    build.main(["--aligned", str(aligned), "--truncated", str(truncated), "--output", str(out), "--keep-unverified"])
    assert len(json.loads(out.read_text())) == 3


def test_truncate_cli_with_the_hf_backend(tmp_path):
    """Smoke: a random tiny judge never says [ENOUGH], so every trace keeps all its sentences."""
    import torch
    from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(str(ASSETS / "tokenizer"))
    torch.manual_seed(0)
    model_dir = tmp_path / "judge"
    config = Qwen2Config(vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64,
                         num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1)
    Qwen2ForCausalLM(config).save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    source = tmp_path / "traces.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in [
        {"question": "q1", "solution": "First. Second. Third.", "answer": "1"},
        {"question": "", "solution": "skipped: no question", "answer": "2"},
        {"question": "q3", "solution": "Only one", "answer": "3"},
    ]))
    out = tmp_path / "truncated.jsonl"
    truncate.main(["--model", str(model_dir), "--input", str(source), "--output", str(out),
                   "--backend", "hf", "--max-new-tokens", "3"])
    rows = [json.loads(line) for line in out.open()]
    assert [r["id"] for r in rows] == [0, 2]
    for row in rows:
        if not row["is_sufficient"]:
            assert row["sufficient_sentences"] == row["total_sentences"] and row["prefix_ratio"] == 1.0
