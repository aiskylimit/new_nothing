import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from distillation.utils import seed_everything
from schema_grounding.inference import model as inference_model
from schema_grounding.inference.checkpoints import (
    DEFAULT_METHODS,
    SUPPORTED_METHODS,
    LastCheckpoint,
    resolve_checkpoint_directory,
    resolve_last_checkpoint,
    select_last_checkpoint,
)
from schema_grounding.inference.data import DatasetSpec, default_dataset_specs, iter_jsonl
from schema_grounding.inference.merge import merge_schema_units
from schema_grounding.inference.model import ModelRunner
from schema_grounding.inference.outputs import ResumableJsonl
from schema_grounding.inference.parsing import parse_selector_label
from schema_grounding.inference.pipeline import (
    InferenceOptions,
    compute_inference_metrics,
    inference_run_complete,
    model_runner_required,
    prepare_run_directory,
    run_dataset_pipeline,
    run_selector_stage,
)
from schema_grounding.inference.prompting import (
    LLAMA3_TEMPLATE_FINGERPRINT,
    LLAMA3_TEMPLATE_NAME,
    QWEN2_5_TEMPLATE_FINGERPRINT,
    QWEN2_5_TEMPLATE_NAME,
    QWEN3_NOTHINK_TEMPLATE_FINGERPRINT,
    QWEN3_NOTHINK_TEMPLATE_NAME,
    PromptTemplates,
    chat_template_metadata,
    qwen_template_metadata,
    render_llama3,
    render_qwen3_nothink,
)
from schema_grounding.selector_labels import format_selector_response
from scripts.infer_two_stage import (
    DEFAULT_INFERENCE_SEEDS,
    build_seed_first_run_groups,
    parse_seeds,
    validate_choices,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_inference_run_complete_requires_every_final_artifact(tmp_path: Path) -> None:
    required_files = (
        "run_config.json",
        "selector_predictions.jsonl",
        "predicted_subschemas.jsonl",
        "generator_predictions.jsonl",
        "metrics.json",
        "manifest.json",
    )
    for filename in required_files:
        (tmp_path / filename).write_text("{}\n", encoding="utf-8")

    assert inference_run_complete(tmp_path)

    (tmp_path / "metrics.json").unlink()
    assert not inference_run_complete(tmp_path)


def write_resume_manifest(
    checkpoint_directory: Path,
    *,
    step: int,
    config_sha256: str = "config-a",
    runtime_sha256: str = "runtime-a",
) -> None:
    (checkpoint_directory / "resume_manifest.json").write_text(
        json.dumps(
            {
                "format_version": 2,
                "global_step": step,
                "config_sha256": config_sha256,
                "runtime_sha256": runtime_sha256,
            }
        ),
        encoding="utf-8",
    )


def node(label: str) -> dict:
    return {
        "id": f"node:{label}",
        "kind": "node",
        "schema": {"label": label, "properties": {"name": "STRING"}},
        "text": f"(:{label} {{ name: STRING }})",
    }


def relation(source: str, relation_type: str, target: str) -> dict:
    return {
        "id": f"relation:{source}|{relation_type}|{target}",
        "kind": "relation",
        "schema": {
            "source": source,
            "type": relation_type,
            "target": target,
            "properties": {},
        },
        "text": f"(:{source})-[:{relation_type}]->(:{target})",
    }


def test_all_methods_include_teacher() -> None:
    assert "teacher_lora" in DEFAULT_METHODS
    assert len(DEFAULT_METHODS) == 13
    assert "teacher_full" in SUPPORTED_METHODS


def test_full_teacher_can_be_requested_explicitly() -> None:
    methods, datasets = validate_choices(SimpleNamespace(methods="teacher_full", datasets="cypherbench"))
    assert methods == ["teacher_full"]
    assert datasets == ["cypherbench"]


def test_selector_label_parser_is_strict() -> None:
    assert format_selector_response("YES") == '{"label": "YES"}'
    assert format_selector_response("NO") == '{"label": "NO"}'
    with pytest.raises(ValueError, match="Unknown selector label"):
        format_selector_response("RELATED")
    assert parse_selector_label('{"label": "YES"}') == "YES"
    assert parse_selector_label(' {"label": "NO"}\n') == "NO"
    assert parse_selector_label('{"label": "yes"}') is None
    assert parse_selector_label('{"label": "YES", "reason": "needed"}') is None
    assert parse_selector_label('{"classification": "YES"}') is None
    # Keep previously trained one-token checkpoints usable during migration.
    assert parse_selector_label(" YES\n") == "YES"
    assert parse_selector_label("NO") == "NO"
    assert parse_selector_label("no") is None
    assert parse_selector_label("The answer is YES") is None
    assert parse_selector_label("RELATED") is None


def test_merge_closes_relationship_endpoints_and_preserves_order() -> None:
    units = [node("A"), node("B"), node("C"), relation("A", "LINKS", "B")]
    result = merge_schema_units(units, ["relation:A|LINKS|B"])
    assert result.sub_schema == {
        "nodes": [units[0]["schema"], units[1]["schema"]],
        "relationships": [units[3]["schema"]],
    }
    assert result.directly_selected_unit_ids == ("relation:A|LINKS|B",)
    assert result.closure_added_node_ids == ("node:A", "node:B")


def test_merge_can_return_the_exact_empty_schema() -> None:
    result = merge_schema_units([node("A")], [])
    assert result.sub_schema == {"nodes": [], "relationships": []}


def test_select_and_resolve_last_local_checkpoint(tmp_path: Path) -> None:
    prefix = "qwen3/sft"
    paths = [f"{prefix}/checkpoint-{step}" for step in (314, 1570, 628)]
    assert select_last_checkpoint(paths, prefix) == (1570, f"{prefix}/checkpoint-1570")

    for path in paths:
        (tmp_path / Path(*path.split("/"))).mkdir(parents=True)

    checkpoint = resolve_last_checkpoint("sft", checkpoint_root=tmp_path)
    assert checkpoint.step == 1570
    assert checkpoint.subfolder == "qwen3/sft/checkpoint-1570"
    assert checkpoint.model_family == "qwen3"
    assert checkpoint.path == tmp_path.resolve() / "qwen3/sft/checkpoint-1570"
    assert len(checkpoint.fingerprint) == 64


def test_checkpoint_fingerprint_changes_when_weights_are_replaced_in_place(tmp_path: Path) -> None:
    checkpoint_directory = tmp_path / "qwen3/sft/checkpoint-10"
    checkpoint_directory.mkdir(parents=True)
    (checkpoint_directory / "adapter_config.json").write_text('{"base": "model"}', encoding="utf-8")
    weights = checkpoint_directory / "adapter_model.safetensors"
    weights.write_bytes(b"old-weights")
    original = resolve_last_checkpoint("sft", checkpoint_root=tmp_path)

    weights.write_bytes(b"new-weights")
    replaced = resolve_last_checkpoint("sft", checkpoint_root=tmp_path)

    assert replaced.path == original.path
    assert replaced.step == original.step
    assert replaced.fingerprint != original.fingerprint


def test_checkpoint_fingerprint_includes_resume_manifest(tmp_path: Path) -> None:
    checkpoint_directory = tmp_path / "qwen3/sft/checkpoint-10"
    checkpoint_directory.mkdir(parents=True)
    (checkpoint_directory / "adapter_config.json").write_text('{}', encoding="utf-8")
    (checkpoint_directory / "adapter_model.safetensors").write_bytes(b"weights")
    write_resume_manifest(checkpoint_directory, step=10, config_sha256="old")
    original = resolve_last_checkpoint("sft", checkpoint_root=tmp_path)

    write_resume_manifest(checkpoint_directory, step=10, config_sha256="new")
    updated = resolve_last_checkpoint("sft", checkpoint_root=tmp_path)

    assert updated.fingerprint != original.fingerprint


def test_last_checkpoint_ignores_higher_incomplete_checkpoint_without_manifest(tmp_path: Path) -> None:
    complete = tmp_path / "qwen3/sft/checkpoint-10"
    incomplete = tmp_path / "qwen3/sft/checkpoint-20"
    complete.mkdir(parents=True)
    incomplete.mkdir(parents=True)
    write_resume_manifest(complete, step=10)

    checkpoint = resolve_last_checkpoint("sft", checkpoint_root=tmp_path)

    assert checkpoint.step == 10
    assert checkpoint.path == complete.resolve()


def test_last_checkpoint_rejects_mixed_training_run_fingerprints(tmp_path: Path) -> None:
    first = tmp_path / "qwen3/sft/checkpoint-10"
    second = tmp_path / "qwen3/sft/checkpoint-20"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    write_resume_manifest(first, step=10, runtime_sha256="runtime-a")
    write_resume_manifest(second, step=20, runtime_sha256="runtime-b")

    with pytest.raises(ValueError, match="Mixed training-run fingerprints"):
        resolve_last_checkpoint("sft", checkpoint_root=tmp_path)


def test_last_checkpoint_rejects_manifest_with_wrong_step(tmp_path: Path) -> None:
    checkpoint_directory = tmp_path / "qwen3/sft/checkpoint-10"
    checkpoint_directory.mkdir(parents=True)
    write_resume_manifest(checkpoint_directory, step=9)

    with pytest.raises(ValueError, match="does not match checkpoint-10"):
        resolve_last_checkpoint("sft", checkpoint_root=tmp_path)


def test_training_output_dirs_match_local_inference_layout() -> None:
    teacher_configs = {
        "qwen3": Path("configs/distillation/teacher_lora_qwen3.yaml"),
        "llama3": Path("configs/distillation/teacher_lora_llama3.yaml"),
        "qwen2.5_coder": Path("configs/distillation/teacher_lora_qwen2.5_coder.yaml"),
    }
    for family, teacher_config in teacher_configs.items():
        output_dirs = {
            path.stem: yaml.safe_load(path.read_text(encoding="utf-8"))["output_dir"]
            for path in (Path("configs") / family).glob("*.yaml")
        }
        output_dirs["teacher_lora"] = yaml.safe_load(teacher_config.read_text(encoding="utf-8"))["output_dir"]

        assert set(output_dirs) == set(DEFAULT_METHODS)
        for method, output_dir in output_dirs.items():
            assert output_dir == f"results/lora/{family}/{method}"

    legacy_qwen_sft = yaml.safe_load(
        Path("configs/distillation/student_sft.yaml").read_text(encoding="utf-8")
    )
    assert legacy_qwen_sft["output_dir"] == "results/lora/qwen3/sft"


def test_default_datasets_use_full_test_inference_artifacts() -> None:
    data_root = Path("/mnt/local/test-cypher-data")
    specs = default_dataset_specs(REPOSITORY_ROOT, data_root=data_root)
    for spec in specs.values():
        assert spec.directory.parent == data_root
        assert spec.generation_test.name == "generation_inference_test.jsonl"
        assert spec.selection_test.name == "selection_inference_test.jsonl"


def test_inference_validates_all_checkpoints_before_preparing_output_directories() -> None:
    source = (REPOSITORY_ROOT / "scripts/infer_two_stage.py").read_text(encoding="utf-8")

    validation = source.index("checkpoint_paths[method] = resolve_checkpoint_directory(checkpoint)")
    output_preparation = source.index("prepare_run_directory(", validation)
    assert validation < output_preparation


def test_cli_rejects_empty_method_or_dataset_lists() -> None:
    with pytest.raises(ValueError, match="at least one method"):
        validate_choices(SimpleNamespace(methods="", datasets="cypherbench"))
    with pytest.raises(ValueError, match="at least one dataset"):
        validate_choices(SimpleNamespace(methods="sft", datasets=""))


def test_default_inference_seeds_and_seed_parser() -> None:
    assert DEFAULT_INFERENCE_SEEDS == (10, 42, 50, 100, 1234)
    assert parse_seeds("10,42,50,100,1234") == [10, 42, 50, 100, 1234]
    with pytest.raises(ValueError, match="duplicates"):
        parse_seeds("10,10")


def test_inference_plan_completes_each_seed_before_the_next(tmp_path: Path) -> None:
    groups = build_seed_first_run_groups(
        methods=["sft", "fkl"],
        dataset_names=["cypherbench", "mind_the_query"],
        seeds=[10, 42],
        output_root=tmp_path,
        options=InferenceOptions(),
    )

    assert [(seed, method) for seed, method, _runs in groups] == [
        (10, "sft"),
        (10, "fkl"),
        (42, "sft"),
        (42, "fkl"),
    ]
    assert [dataset for _seed, _method, runs in groups for dataset, _path, _options in runs] == [
        "cypherbench",
        "mind_the_query",
        "cypherbench",
        "mind_the_query",
        "cypherbench",
        "mind_the_query",
        "cypherbench",
        "mind_the_query",
    ]
    assert [run_options.seed for _seed, _method, runs in groups for _dataset, _path, run_options in runs] == [
        10,
        10,
        10,
        10,
        42,
        42,
        42,
        42,
    ]


def test_local_checkpoint_accepts_adapter_weights(tmp_path: Path) -> None:
    checkpoint = LastCheckpoint(str(tmp_path), "qwen3", "sft", 1570, "qwen3/sft/checkpoint-1570")
    adapter = checkpoint.path
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")

    assert resolve_checkpoint_directory(checkpoint) == adapter


def test_local_checkpoint_accepts_full_model_weights(tmp_path: Path) -> None:
    checkpoint = LastCheckpoint(
        str(tmp_path), "qwen2.5_coder", "sft", 10, "qwen2.5_coder/sft/checkpoint-10"
    )
    model_dir = checkpoint.path
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"weights")

    assert resolve_checkpoint_directory(checkpoint) == model_dir


def test_prompt_messages_match_training_format() -> None:
    templates = PromptTemplates.from_repository(REPOSITORY_ROOT)
    generator = templates.generator_messages(
        "Who?", {"nodes": [{"label": "Person", "properties": {}}], "relationships": []}
    )
    selector = templates.selector_messages("Who?", "(:Person)")
    assert (
        generator[0]["content"]
        == (REPOSITORY_ROOT / "prompts/generator/system_prompt.txt").read_text(encoding="utf-8").strip()
    )
    assert "QUESTION:\nWho?" in generator[1]["content"]
    assert '"relationships": []' in generator[1]["content"]
    assert (
        selector[0]["content"]
        == (REPOSITORY_ROOT / "prompts/selector/system_prompt.txt").read_text(encoding="utf-8").strip()
    )
    assert "SCHEMA UNIT:\n(:Person)" in selector[1]["content"]
    assert selector[1]["content"].endswith("Classify the schema unit and return only the required JSON object.")
    assert '{"label": "YES"}' in selector[0]["content"]
    assert "`" not in selector[0]["content"]


def test_qwen3_nothink_renderer_exactly_matches_llamafactory_chatml() -> None:
    rendered = render_qwen3_nothink(
        [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Question"},
        ]
    )

    assert rendered == (
        "<|im_start|>system\nSystem<|im_end|>\n"
        "<|im_start|>user\nQuestion<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    assert "<think>" not in rendered
    assert QWEN3_NOTHINK_TEMPLATE_NAME == "llamafactory:qwen3_nothink"
    assert len(QWEN3_NOTHINK_TEMPLATE_FINGERPRINT) == 64


def test_qwen2_5_uses_qwen_template_with_the_same_chatml_serialization() -> None:
    assert qwen_template_metadata("qwen2.5_coder") == {
        "name": QWEN2_5_TEMPLATE_NAME,
        "fingerprint": QWEN2_5_TEMPLATE_FINGERPRINT,
    }
    assert QWEN2_5_TEMPLATE_NAME == "llamafactory:qwen"
    assert QWEN2_5_TEMPLATE_FINGERPRINT == QWEN3_NOTHINK_TEMPLATE_FINGERPRINT


def test_llama3_inference_records_the_llamafactory_template() -> None:
    assert chat_template_metadata("llama3") == {
        "name": LLAMA3_TEMPLATE_NAME,
        "fingerprint": LLAMA3_TEMPLATE_FINGERPRINT,
    }
    assert LLAMA3_TEMPLATE_NAME == "llamafactory:llama3"
    assert len(LLAMA3_TEMPLATE_FINGERPRINT) == 64

    assert render_llama3(
        [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Question"},
        ],
        bos_token="<|begin_of_text|>",
    ) == (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "System<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        "Question<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def test_model_runner_tokenizes_rendered_llamafactory_template() -> None:
    class RecordingTokenizer:
        messages = None
        kwargs = None

        def apply_chat_template(self, messages, **kwargs) -> list[int]:
            self.messages = messages
            self.kwargs = kwargs
            return [1, 2, 3]

    tokenizer = RecordingTokenizer()
    runner = ModelRunner(model=None, tokenizer=tokenizer, device=torch.device("cpu"))

    assert runner.prompt_length([{"role": "user", "content": "Question"}]) == 3
    assert tokenizer.messages == [{"role": "user", "content": "Question"}]
    assert tokenizer.kwargs == {
        "tokenize": True,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }


def test_model_runner_uses_explicit_nothink_template_for_qwen3() -> None:
    class RecordingTokenizer:
        rendered = None

        def encode(self, rendered, **kwargs) -> list[int]:
            self.rendered = rendered
            assert kwargs == {"add_special_tokens": False}
            return [1, 2, 3]

    tokenizer = RecordingTokenizer()
    runner = ModelRunner(
        model=None,
        tokenizer=tokenizer,
        device=torch.device("cpu"),
        model_family="qwen3",
    )

    assert runner.prompt_length([{"role": "user", "content": "Question"}]) == 3
    assert tokenizer.rendered == (
        "<|im_start|>user\nQuestion<|im_end|>\n<|im_start|>assistant\n"
    )
    assert "<think>" not in tokenizer.rendered


class FakeRunner:
    model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=4096))

    def prompt_length(self, messages) -> int:
        return sum(len(message["content"].split()) for message in messages)

    def generate(self, conversations, *, max_new_tokens, **generation_kwargs):
        is_selector = "relevance classifier" in conversations[0][0]["content"]
        expected_kwargs = (
            {
                "do_sample": False,
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": 0,
                "num_beams": 1,
            }
            if is_selector
            else {
                "do_sample": True,
                "temperature": 0.5,
                "top_p": 0.95,
                "top_k": 0,
                "num_beams": 1,
            }
        )
        assert generation_kwargs == expected_kwargs
        assert max_new_tokens == (16 if is_selector else 256)
        outputs = []
        for messages in conversations:
            system = messages[0]["content"]
            user = messages[1]["content"]
            if "relevance classifier" in system:
                label = "YES" if "(:A " in user or "[:LINKS]" in user else "NO"
                outputs.append(json.dumps({"label": label}))
            else:
                # Deliberately omit the closing JSON brace to exercise the
                # inference-time recovery used for real model generations.
                outputs.append('{"cypher": "MATCH (n:A) RETURN n.name"')
        return outputs


class CountingRunner(FakeRunner):
    def __init__(self) -> None:
        self.generated = 0

    def generate(self, conversations, *, max_new_tokens, **generation_kwargs):
        self.generated += len(conversations)
        return super().generate(conversations, max_new_tokens=max_new_tokens, **generation_kwargs)


def test_inference_options_rejects_negative_top_k() -> None:
    with pytest.raises(ValueError, match="top_k must be a non-negative integer"):
        InferenceOptions(top_k=-1).validate()


def test_inference_options_requires_positive_selector_response_budget() -> None:
    with pytest.raises(ValueError, match="selector_max_new_tokens must be a positive integer"):
        InferenceOptions(selector_max_new_tokens=0).validate()


def test_model_runner_honors_base_revision_and_ignores_incomplete_local_tokenizer(monkeypatch, tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    calls: dict[str, tuple] = {}

    class DummyModel:
        def to(self, device):
            calls["device"] = (device,)
            return self

        def eval(self):
            return self

    monkeypatch.setattr(
        inference_model.PeftConfig,
        "from_pretrained",
        lambda path: SimpleNamespace(base_model_name_or_path="owner/base", revision="base-commit"),
    )

    def fake_tokenizer(source, **kwargs):
        calls["tokenizer"] = (source, kwargs)
        return SimpleNamespace(pad_token_id=0, padding_side="right")

    def fake_base_model(source, **kwargs):
        calls["model"] = (source, kwargs)
        return DummyModel()

    monkeypatch.setattr(inference_model.AutoTokenizer, "from_pretrained", fake_tokenizer)
    monkeypatch.setattr(inference_model.AutoModelForCausalLM, "from_pretrained", fake_base_model)
    monkeypatch.setattr(inference_model.PeftModel, "from_pretrained", lambda model, path: model)

    ModelRunner.from_adapter(adapter, device="cpu", merge_adapter=False)

    assert calls["tokenizer"] == ("owner/base", {"revision": "base-commit", "use_fast": True})
    assert calls["model"][0] == "owner/base"
    assert calls["model"][1]["revision"] == "base-commit"


def test_model_runner_prefers_manifest_base_commit_over_moving_adapter_revision(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "resume_manifest.json").write_text(
        json.dumps(
            {
                "format_version": 2,
                "runtime": {
                    "student_model": {
                        "name_or_path": "owner/base",
                        "commit_hash": "immutable-commit",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    assert inference_model._pinned_base_revision(adapter, "owner/base", "main") == "immutable-commit"


def test_model_runner_rejects_manifest_for_another_base_model(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "resume_manifest.json").write_text(
        json.dumps(
            {
                "format_version": 2,
                "runtime": {
                    "student_model": {
                        "name_or_path": "owner/wrong-base",
                        "commit_hash": "immutable-commit",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="does not match adapter base model"):
        inference_model._pinned_base_revision(adapter, "owner/base", "main")


def test_model_runner_loads_full_checkpoint_without_peft(monkeypatch, tmp_path: Path) -> None:
    checkpoint = tmp_path / "full"
    checkpoint.mkdir()
    calls: dict[str, str] = {}

    class DummyModel:
        def to(self, device):
            calls["device"] = str(device)
            return self

        def eval(self):
            return self

    def fake_tokenizer(source, **kwargs):
        calls["tokenizer"] = str(source)
        return SimpleNamespace(pad_token_id=0, padding_side="right")

    def fake_model(source, **kwargs):
        calls["model"] = str(source)
        return DummyModel()

    monkeypatch.setattr(inference_model.AutoTokenizer, "from_pretrained", fake_tokenizer)
    monkeypatch.setattr(inference_model.AutoModelForCausalLM, "from_pretrained", fake_model)
    monkeypatch.setattr(
        inference_model.PeftConfig,
        "from_pretrained",
        lambda path: pytest.fail("PEFT must not load for a full checkpoint"),
    )

    ModelRunner.from_checkpoint(checkpoint, device="cpu")

    assert calls == {"tokenizer": str(checkpoint), "model": str(checkpoint), "device": "cpu"}


def test_model_runner_remembers_safe_batch_size_after_oom(monkeypatch) -> None:
    runner = ModelRunner(model=None, tokenizer=None, device=torch.device("cpu"))
    attempted_batch_sizes: list[int] = []

    def fake_generate_batch(conversations, *, max_new_tokens, **generation_kwargs):
        assert generation_kwargs == {
            "do_sample": True,
            "temperature": 0.5,
            "top_p": 0.95,
            "top_k": 0,
            "num_beams": 1,
        }
        attempted_batch_sizes.append(len(conversations))
        if len(conversations) > 2:
            raise torch.cuda.OutOfMemoryError("simulated OOM")
        return ["ok"] * len(conversations)

    monkeypatch.setattr(runner, "_generate_batch", fake_generate_batch)
    conversations = [[{"role": "user", "content": str(index)}] for index in range(8)]

    assert runner.generate(conversations, max_new_tokens=8) == ["ok"] * 8
    attempts_after_first_call = len(attempted_batch_sizes)
    assert runner.safe_batch_sizes[8] == 2
    assert runner.generate(conversations, max_new_tokens=8) == ["ok"] * 8
    assert all(size <= 2 for size in attempted_batch_sizes[attempts_after_first_call:])


def test_model_runner_restores_rng_before_oom_batch_split(monkeypatch) -> None:
    def fake_generate_batch(conversations, **kwargs):
        del kwargs
        values = torch.rand(len(conversations)).tolist()
        if len(conversations) > 2:
            raise torch.cuda.OutOfMemoryError("simulated OOM")
        return [str(value) for value in values]

    conversations = [[{"role": "user", "content": str(index)}] for index in range(4)]
    recovered = ModelRunner(model=None, tokenizer=None, device=torch.device("cpu"))
    monkeypatch.setattr(recovered, "_generate_batch", fake_generate_batch)
    seed_everything(77, rank_offset=False)
    recovered_outputs = recovered.generate(conversations, max_new_tokens=8)

    direct = ModelRunner(model=None, tokenizer=None, device=torch.device("cpu"), safe_batch_sizes={8: 2})
    monkeypatch.setattr(direct, "_generate_batch", fake_generate_batch)
    seed_everything(77, rank_offset=False)
    direct_outputs = direct.generate(conversations, max_new_tokens=8)

    assert recovered_outputs == direct_outputs


def test_resumable_jsonl_restores_rng_and_rolls_back_uncommitted_batch_tail(tmp_path: Path) -> None:
    output = ResumableJsonl(tmp_path / "generator_predictions.jsonl")
    seed_everything(91, rank_offset=False)
    with output.open_append() as handle:
        handle.write('{"id":"row-1"}\n{"id":"row-2"}\n')
        output.checkpoint_rng_progress(handle, 2, "row-2")
    expected_next = torch.rand(4)
    with output.open_append() as handle:
        handle.write('{"id":"uncommitted-row"}\n')

    seed_everything(999, rank_offset=False)
    assert output.restore_rng_progress() == (2, "row-2")
    torch.testing.assert_close(torch.rand(4), expected_next)
    assert [row["id"] for row in iter_jsonl(output.partial_path)] == ["row-1", "row-2"]


def test_resumable_jsonl_restarts_uncommitted_legacy_partial(tmp_path: Path) -> None:
    output = ResumableJsonl(tmp_path / "generator_predictions.jsonl")
    output.partial_path.write_text('{"id":"uncommitted"}\n', encoding="utf-8")

    assert output.restore_rng_progress() == (0, None)
    assert not output.partial_path.exists()


def test_end_to_end_pipeline_with_fake_model(tmp_path: Path) -> None:
    data = tmp_path / "benchmark"
    units = [node("A"), node("B"), relation("A", "LINKS", "B")]
    generation = {
        "id": "example-1",
        "source": "fixture",
        "split": "test",
        "graph": "graph",
        "schema_id": "schema-1",
        "question": "Find A",
        "cypher": "MATCH (n:A) RETURN n.name",
        "sub_schema": {
            "nodes": [units[0]["schema"], units[1]["schema"]],
            "relationships": [units[2]["schema"]],
        },
    }
    selection = [
        {
            "id": f"example-1:{unit['id']}",
            "example_id": "example-1",
            "schema_id": "schema-1",
            "unit_id": unit["id"],
            "unit_type": unit["kind"],
            "unit": unit,
            "question": "Find A",
            "label": int(unit["id"] != "node:B"),
        }
        for unit in units
    ]
    write_jsonl(data / "generation_test.jsonl", [generation])
    write_jsonl(data / "selection_test.jsonl", selection)
    output = tmp_path / "output"
    checkpoint = LastCheckpoint("results", "qwen3", "sft", 7, "qwen3/sft/checkpoint-7")
    manifest = run_dataset_pipeline(
        method="sft",
        checkpoint=checkpoint,
        spec=DatasetSpec("fixture", data),
        runner=FakeRunner(),
        templates=PromptTemplates.from_repository(REPOSITORY_ROOT),
        output_directory=output,
        options=InferenceOptions(selector_batch_size=2, generator_batch_size=1),
    )
    assert manifest["method"] == "sft"
    run_config = json.loads((output / "run_config.json").read_text(encoding="utf-8"))
    assert run_config["chat_template"] == {
        "name": QWEN3_NOTHINK_TEMPLATE_NAME,
        "fingerprint": QWEN3_NOTHINK_TEMPLATE_FINGERPRINT,
    }
    assert run_config["selector_protocol"] == {
        "positive_label": "YES",
        "negative_label": "NO",
        "output_format": {"label": "YES|NO"},
        "do_sample": False,
        "num_beams": 1,
    }
    prediction = list(iter_jsonl(output / "generator_predictions.jsonl"))[0]
    assert prediction["predicted_cypher"] == generation["cypher"]
    assert prediction["predicted_sub_schema"] == generation["sub_schema"]
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["selector"]["accuracy"] == pytest.approx(100.0)
    assert metrics["generator"]["generator_exact_match"] == pytest.approx(100.0)


def test_pipeline_reuses_completed_stages(tmp_path: Path) -> None:
    test_end_to_end_pipeline_with_fake_model(tmp_path)
    output = tmp_path / "output"
    checkpoint = LastCheckpoint("results", "qwen3", "sft", 7, "qwen3/sft/checkpoint-7")
    manifest = run_dataset_pipeline(
        method="sft",
        checkpoint=checkpoint,
        spec=DatasetSpec("fixture", tmp_path / "benchmark"),
        runner=None,
        templates=PromptTemplates.from_repository(REPOSITORY_ROOT),
        output_directory=output,
        options=InferenceOptions(selector_batch_size=2, generator_batch_size=1),
    )
    assert all(stage["status"] == "reused" for stage in manifest["stages"].values())
    assert not model_runner_required(output)


def test_preflight_rejects_downstream_output_without_its_dependency(tmp_path: Path) -> None:
    data = tmp_path / "benchmark"
    write_jsonl(data / "generation_test.jsonl", [{"id": "example-1"}])
    write_jsonl(data / "selection_test.jsonl", [{"id": "unit-1"}])
    output = tmp_path / "output"
    write_jsonl(output / "predicted_subschemas.jsonl", [{"id": "example-1"}])

    write_jsonl(output / "run_config.json", [{}])

    with pytest.raises(ValueError, match="sub-schema output exists without completed selector_predictions"):
        prepare_run_directory(
            method="sft",
            checkpoint=LastCheckpoint("results", "qwen3", "sft", 7, "qwen3/sft/checkpoint-7"),
            spec=DatasetSpec("fixture", data),
            templates=PromptTemplates.from_repository(REPOSITORY_ROOT),
            output_directory=output,
            options=InferenceOptions(),
        )


def test_preflight_rejects_orphaned_outputs_without_run_config(tmp_path: Path) -> None:
    data = tmp_path / "benchmark"
    write_jsonl(data / "generation_test.jsonl", [{"id": "example-1"}])
    write_jsonl(data / "selection_test.jsonl", [{"id": "unit-1"}])
    output = tmp_path / "output"
    write_jsonl(output / "selector_predictions.jsonl", [{"id": "unit-1"}])

    with pytest.raises(ValueError, match="without run_config.json"):
        prepare_run_directory(
            method="sft",
            checkpoint=LastCheckpoint("results", "qwen3", "sft", 7, "qwen3/sft/checkpoint-7"),
            spec=DatasetSpec("fixture", data),
            templates=PromptTemplates.from_repository(REPOSITORY_ROOT),
            output_directory=output,
            options=InferenceOptions(),
        )


def test_pipeline_rejects_stale_outputs_from_another_checkpoint(tmp_path: Path) -> None:
    test_end_to_end_pipeline_with_fake_model(tmp_path)
    changed_checkpoint = LastCheckpoint("results", "qwen3", "sft", 8, "qwen3/sft/checkpoint-8")
    with pytest.raises(ValueError, match="different configuration"):
        run_dataset_pipeline(
            method="sft",
            checkpoint=changed_checkpoint,
            spec=DatasetSpec("fixture", tmp_path / "benchmark"),
            runner=FakeRunner(),
            templates=PromptTemplates.from_repository(REPOSITORY_ROOT),
            output_directory=tmp_path / "output",
            options=InferenceOptions(selector_batch_size=2, generator_batch_size=1),
        )


def test_pipeline_rejects_replaced_weights_at_same_checkpoint_path(tmp_path: Path) -> None:
    test_end_to_end_pipeline_with_fake_model(tmp_path)
    changed_checkpoint = LastCheckpoint(
        "results",
        "qwen3",
        "sft",
        7,
        "qwen3/sft/checkpoint-7",
        fingerprint="replaced-weights",
    )

    with pytest.raises(ValueError, match="different configuration"):
        run_dataset_pipeline(
            method="sft",
            checkpoint=changed_checkpoint,
            spec=DatasetSpec("fixture", tmp_path / "benchmark"),
            runner=FakeRunner(),
            templates=PromptTemplates.from_repository(REPOSITORY_ROOT),
            output_directory=tmp_path / "output",
            options=InferenceOptions(selector_batch_size=2, generator_batch_size=1),
        )


def test_pipeline_rejects_stale_outputs_after_input_content_changes(tmp_path: Path) -> None:
    test_end_to_end_pipeline_with_fake_model(tmp_path)
    generation_path = tmp_path / "benchmark" / "generation_test.jsonl"
    generation = list(iter_jsonl(generation_path))[0]
    generation["question"] = "Changed question"
    write_jsonl(generation_path, [generation])

    checkpoint = LastCheckpoint("results", "qwen3", "sft", 7, "qwen3/sft/checkpoint-7")
    with pytest.raises(ValueError, match="different configuration"):
        run_dataset_pipeline(
            method="sft",
            checkpoint=checkpoint,
            spec=DatasetSpec("fixture", tmp_path / "benchmark"),
            runner=FakeRunner(),
            templates=PromptTemplates.from_repository(REPOSITORY_ROOT),
            output_directory=tmp_path / "output",
            options=InferenceOptions(selector_batch_size=2, generator_batch_size=1),
        )


def test_pipeline_rejects_stale_outputs_after_prompt_changes(tmp_path: Path) -> None:
    test_end_to_end_pipeline_with_fake_model(tmp_path)
    templates = PromptTemplates.from_repository(REPOSITORY_ROOT)
    changed_templates = PromptTemplates(
        generator_system=templates.generator_system + " Changed.",
        generator_user=templates.generator_user,
        selector_system=templates.selector_system,
        selector_user=templates.selector_user,
    )
    checkpoint = LastCheckpoint("results", "qwen3", "sft", 7, "qwen3/sft/checkpoint-7")

    with pytest.raises(ValueError, match="different configuration"):
        run_dataset_pipeline(
            method="sft",
            checkpoint=checkpoint,
            spec=DatasetSpec("fixture", tmp_path / "benchmark"),
            runner=FakeRunner(),
            templates=changed_templates,
            output_directory=tmp_path / "output",
            options=InferenceOptions(selector_batch_size=2, generator_batch_size=1),
        )


def test_invalid_selector_predictions_are_not_counted_as_true_negatives(tmp_path: Path) -> None:
    data = tmp_path / "benchmark"
    selection = [
        {"id": "negative", "label": 0},
        {"id": "positive", "label": 1},
        {"id": "unlabeled"},
    ]
    predictions = [
        {"id": "negative", "predicted_label": "INVALID"},
        {"id": "positive", "predicted_label": "INVALID"},
        {"id": "unlabeled", "predicted_label": "INVALID"},
    ]
    generator_predictions = [
        {
            "id": "example-1",
            "predicted_cypher": "RETURN 1",
            "reference_cypher": "WRONG EMBEDDED REFERENCE",
            "closure_added_node_ids": [],
            "predicted_sub_schema": {"nodes": [], "relationships": []},
        }
    ]
    write_jsonl(data / "selection_test.jsonl", selection)
    write_jsonl(data / "generation_test.jsonl", [{"id": "example-1", "cypher": "RETURN 1"}])
    write_jsonl(tmp_path / "selector_predictions.jsonl", predictions)
    write_jsonl(tmp_path / "generator_predictions.jsonl", generator_predictions)

    metrics = compute_inference_metrics(
        DatasetSpec("fixture", data),
        tmp_path / "selector_predictions.jsonl",
        tmp_path / "generator_predictions.jsonl",
    )

    assert metrics["selector"]["count"] == 2
    assert metrics["selector"]["inference_count"] == 3
    assert metrics["selector"]["unlabeled_count"] == 1
    assert metrics["selector"]["accuracy"] == 0.0
    assert metrics["selector"]["true_negative"] == 0
    assert metrics["selector"]["false_negative"] == 1
    assert metrics["selector"]["invalid"] == 2
    assert metrics["selector"]["inference_invalid"] == 3
    assert metrics["generator"]["generator_exact_match"] == 100.0


def test_metrics_reject_misaligned_selector_prediction_ids(tmp_path: Path) -> None:
    data = tmp_path / "benchmark"
    write_jsonl(data / "selection_test.jsonl", [{"id": "expected", "label": 0}])
    write_jsonl(data / "generation_test.jsonl", [{"id": "example-1", "cypher": "RETURN 1"}])
    write_jsonl(tmp_path / "selector_predictions.jsonl", [{"id": "other", "predicted_label": "NO"}])
    write_jsonl(
        tmp_path / "generator_predictions.jsonl",
        [
            {
                "id": "example-1",
                "predicted_cypher": "RETURN 1",
                "closure_added_node_ids": [],
                "predicted_sub_schema": {"nodes": [], "relationships": []},
            }
        ],
    )

    with pytest.raises(ValueError, match="misalignment"):
        compute_inference_metrics(
            DatasetSpec("fixture", data),
            tmp_path / "selector_predictions.jsonl",
            tmp_path / "generator_predictions.jsonl",
        )


def test_rejected_example_zeros_selector_but_generator_is_scored_normally(tmp_path: Path) -> None:
    data = tmp_path / "benchmark"
    write_jsonl(
        data / "selection_test.jsonl",
        [
            {"id": "accepted-unit", "example_id": "accepted", "label": 1},
            {"id": "rejected-unit", "example_id": "rejected"},
        ],
    )
    write_jsonl(
        data / "generation_test.jsonl",
        [
            {
                "id": "accepted",
                "cypher": "RETURN 1",
                "gold_subschema_available": True,
            },
            {
                "id": "rejected",
                "cypher": "RETURN",
                "gold_subschema_available": False,
            },
        ],
    )
    write_jsonl(
        tmp_path / "selector_predictions.jsonl",
        [
            {"id": "accepted-unit", "predicted_label": "YES"},
            {"id": "rejected-unit", "predicted_label": "YES"},
        ],
    )
    write_jsonl(
        tmp_path / "generator_predictions.jsonl",
        [
            {
                "id": "accepted",
                "predicted_cypher": "RETURN 1",
                "closure_added_node_ids": [],
                "predicted_sub_schema": {"nodes": [], "relationships": []},
            },
            {
                "id": "rejected",
                "predicted_cypher": "RETURN",
                "closure_added_node_ids": [],
                "predicted_sub_schema": {"nodes": [], "relationships": []},
            },
        ],
    )

    metrics = compute_inference_metrics(
        DatasetSpec("fixture", data),
        tmp_path / "selector_predictions.jsonl",
        tmp_path / "generator_predictions.jsonl",
    )

    assert metrics["generator"]["generator_count"] == 2.0
    assert metrics["selector"]["accuracy"] == 50.0
    assert metrics["selector"]["strict_accuracy"] == 100.0
    assert metrics["selector"]["zero_scored_example_count"] == 1
    assert metrics["generator"]["generator_exact_match"] == 100.0
    assert metrics["generator"]["generator_rouge1"] == 100.0
    assert metrics["generator"]["generator_rouge2"] == 100.0
    assert metrics["generator"]["generator_rougeL"] == 100.0


def test_selector_stage_resumes_a_contiguous_partial_file(tmp_path: Path) -> None:
    data = tmp_path / "benchmark"
    units = [node("A"), node("B"), relation("A", "LINKS", "B")]
    selection = [
        {
            "id": f"example-1:{unit['id']}",
            "example_id": "example-1",
            "schema_id": "schema-1",
            "unit_id": unit["id"],
            "unit_type": unit["kind"],
            "unit": unit,
            "question": "Find A",
            "label": 1,
        }
        for unit in units
    ]
    write_jsonl(data / "selection_test.jsonl", selection)
    output = tmp_path / "selector_predictions.jsonl"
    partial = output.with_suffix(".jsonl.partial")
    write_jsonl(
        partial,
        [
            {
                "id": selection[0]["id"],
                "example_id": "example-1",
                "schema_id": "schema-1",
                "unit_id": "node:A",
                "unit_type": "node",
                "predicted_label": "YES",
                "valid": True,
                "raw_output": "YES",
            }
        ],
    )
    resumable_output = ResumableJsonl(output)
    seed_everything(42, rank_offset=False)
    with partial.open("a", encoding="utf-8") as handle:
        resumable_output.checkpoint_rng_progress(handle, 1, selection[0]["id"])
    runner = CountingRunner()
    result = run_selector_stage(
        DatasetSpec("fixture", data),
        runner,
        PromptTemplates.from_repository(REPOSITORY_ROOT),
        output,
        InferenceOptions(selector_batch_size=2),
    )
    assert result["resumed_rows"] == 1
    assert runner.generated == 2
    assert len(list(iter_jsonl(output))) == 3
    assert not partial.exists()
