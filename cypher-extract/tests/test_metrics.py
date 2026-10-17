import numpy as np

from distillation.metrics import compute_task_metrics, decode_task_outputs, extract_cypher


def test_extract_cypher_from_json_and_fence() -> None:
    assert extract_cypher('{"cypher": "MATCH (n) RETURN n"}') == "MATCH (n) RETURN n"
    assert extract_cypher("```cypher\nMATCH (n) RETURN n\n```") == "MATCH (n) RETURN n"


def test_extract_cypher_from_malformed_or_truncated_json() -> None:
    assert extract_cypher('{"cypher": "MATCH (n {name: \\\'Neo\\\'}) RETURN n"}') == (
        "MATCH (n {name: 'Neo'}) RETURN n"
    )
    assert extract_cypher('{"cypher": "MATCH (n) RETURN n"') == "MATCH (n) RETURN n"
    assert extract_cypher('{"cypher": "MATCH (n) RETURN n\n}') == "MATCH (n) RETURN n"


def test_compute_mixed_task_metrics() -> None:
    metrics = compute_task_metrics(
        [
            '{"label": "YES"}',
            '{"label": "YES"}',
            '{"cypher":"MATCH (n) RETURN n"}',
            '{"cypher":"MATCH (n) RETURN n.name"}',
        ],
        [
            '{"label": "YES"}',
            '{"label": "NO"}',
            '{"cypher":"MATCH (n)  RETURN n;"}',
            '{"cypher":"MATCH (n) RETURN n.age"}',
        ],
    )

    assert metrics["selector_count"] == 2
    assert metrics["selector_accuracy"] == 50.0
    assert metrics["generator_count"] == 2
    assert metrics["generator_exact_match"] == 50.0
    assert 0.0 < metrics["generator_rouge1"] < 100.0
    assert 0.0 < metrics["generator_rouge2"] < 100.0
    assert 0.0 < metrics["generator_rougeL"] < 100.0


def test_selector_training_metric_uses_strict_inference_parser() -> None:
    metrics = compute_task_metrics(["The answer is YES"], ['{"label": "YES"}'])

    assert metrics["selector_count"] == 1
    assert metrics["selector_accuracy"] == 0.0


def test_decode_task_outputs_replaces_ignored_ids_before_decoding() -> None:
    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 2

        @staticmethod
        def batch_decode(rows, skip_special_tokens):
            assert skip_special_tokens is True
            return [" ".join(str(token) for token in row if token != 0) for row in rows]

    predictions, references = decode_task_outputs(
        Tokenizer(),
        np.array([[4, 5, -100]]),
        np.array([[-100, 6, 7]]),
    )

    assert predictions == ["4 5"]
    assert references == ["6 7"]
