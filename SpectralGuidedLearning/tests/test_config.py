import pytest

from sgl.config import ConfigError, apply_overrides, deep_merge, interpolate, load_config, parse_yaml


def write(path, text):
    path.write_text(text)
    return path


def test_deep_merge_recurses_into_mappings_and_replaces_everything_else():
    base = {"a": {"x": 1, "y": [1, 2]}, "b": 1}
    merged = deep_merge(base, {"a": {"y": [3]}, "c": 2})
    assert merged == {"a": {"x": 1, "y": [3]}, "b": 1, "c": 2}
    assert base == {"a": {"x": 1, "y": [1, 2]}, "b": 1}  # inputs untouched


def test_defaults_chain_merges_in_order_with_the_file_last(tmp_path):
    (tmp_path / "sub").mkdir()
    write(tmp_path / "sub" / "a.yaml", "x: 1\nnested: {k: a, keep: true}\n")
    write(tmp_path / "b.yaml", "defaults: [sub/a.yaml]\nx: 2\nnested: {k: b}\n")
    write(tmp_path / "top.yaml", "defaults: [b.yaml]\ny: ${x}\n")
    assert load_config(tmp_path / "top.yaml") == {"x": 2, "nested": {"k": "b", "keep": True}, "y": 2}


def test_defaults_cycle_is_reported(tmp_path):
    write(tmp_path / "a.yaml", "defaults: [b.yaml]\n")
    write(tmp_path / "b.yaml", "defaults: [a.yaml]\n")
    with pytest.raises(ConfigError, match="cycle"):
        load_config(tmp_path / "a.yaml")


def test_interpolation_keeps_types_for_whole_references_and_formats_embedded_ones():
    config = interpolate({
        "n": 3, "flag": False, "gpus": [0, 1], "name": "arm", "track": "t",
        "run": "${name}-${track}", "count": "${n}", "list": "${gpus}", "text": "x${flag}", "nested": "${run}/ckpt",
    })
    assert config["run"] == "arm-t"
    assert config["count"] == 3 and config["list"] == [0, 1]
    assert config["text"] == "xfalse"
    assert config["nested"] == "arm-t/ckpt"


def test_interpolation_errors(monkeypatch):
    with pytest.raises(ConfigError, match="unknown reference"):
        interpolate({"a": "${missing}"})
    with pytest.raises(ConfigError, match="cannot embed"):
        interpolate({"l": [1], "a": "x${l}"})
    with pytest.raises(ConfigError, match="too deep"):
        interpolate({"a": "${b}", "b": "${a}"})
    monkeypatch.delenv("SGL_TEST_VAR", raising=False)
    assert interpolate({"a": "${env:SGL_TEST_VAR,fallback}"})["a"] == "fallback"
    with pytest.raises(ConfigError, match="not set"):
        interpolate({"a": "${env:SGL_TEST_VAR}"})
    monkeypatch.setenv("SGL_TEST_VAR", "set")
    assert interpolate({"a": "${env:SGL_TEST_VAR}"})["a"] == "set"


def test_overrides_create_nested_keys_and_parse_values():
    config = apply_overrides({"a": {"b": 1}}, ["a.b=2", "a.c.d=[1, 2]", "e=hello", "f="])
    assert config == {"a": {"b": 2, "c": {"d": [1, 2]}}, "e": "hello", "f": ""}
    with pytest.raises(ConfigError):
        apply_overrides({"a": 1}, ["a.b=2"])
    with pytest.raises(ConfigError):
        apply_overrides({}, ["no-equals-sign"])


def test_only_true_and_false_are_booleans():
    # YAML 1.1 would turn these into False/True and silently drop e.g. `--think-prefix off`
    assert parse_yaml("a: off\nb: no\nc: on\nd: yes\ne: true\nf: False") == {
        "a": "off", "b": "no", "c": "on", "d": "yes", "e": True, "f": False,
    }
    assert apply_overrides({}, ["think_prefix=off"]) == {"think_prefix": "off"}
