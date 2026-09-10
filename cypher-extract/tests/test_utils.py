import random
from pathlib import Path

import numpy as np
import pytest
import torch

from distillation.utils import (
    all_gather_tensor,
    capture_rng_state,
    get_rank,
    hf_paths,
    parse_hf_path,
    resolve_hf_path,
    restore_rng_state,
    save_rank,
    seed_everything,
)


def test_parse_hf_path_with_subdirectory() -> None:
    assert parse_hf_path("hf://owner/repository/a/b") == ("owner/repository", "a/b")


def test_parse_hf_path_rejects_incomplete_uri() -> None:
    with pytest.raises(ValueError, match="Expected"):
        parse_hf_path("hf://owner")


def test_resolve_hf_subdirectory(monkeypatch, tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    subdirectory = snapshot / "adapter" / "100"
    subdirectory.mkdir(parents=True)

    def fake_snapshot_download(**kwargs) -> str:
        assert kwargs["repo_id"] == "owner/repo"
        assert kwargs["allow_patterns"] == ["adapter/100/*", "adapter/100/**"]
        return str(snapshot)

    resolve_hf_path.cache_clear()
    monkeypatch.setattr(hf_paths, "snapshot_download", fake_snapshot_download)
    assert resolve_hf_path("hf://owner/repo/adapter/100") == str(subdirectory)
    resolve_hf_path.cache_clear()


def test_single_process_gather_semantics() -> None:
    tensor = torch.tensor([[1, 2]])
    assert all_gather_tensor(tensor, operation="cat") is tensor
    torch.testing.assert_close(all_gather_tensor(tensor, operation="stack"), tensor.unsqueeze(0))


def test_seed_everything_is_repeatable() -> None:
    seed_everything(12, rank_offset=False)
    first = torch.rand(3)
    seed_everything(12, rank_offset=False)
    second = torch.rand(3)
    torch.testing.assert_close(first, second)


def test_capture_and_restore_rng_state_covers_python_numpy_and_torch() -> None:
    seed_everything(19, rank_offset=False)
    state = capture_rng_state()
    expected = (random.random(), np.random.random(), torch.rand(3))

    seed_everything(999, rank_offset=False)
    restore_rng_state(state)
    actual = (random.random(), np.random.random(), torch.rand(3))

    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    torch.testing.assert_close(actual[2], expected[2])


def test_rank_is_available_from_torchrun_environment_before_process_group(monkeypatch) -> None:
    monkeypatch.setenv("RANK", "3")
    assert get_rank() == 3
    assert seed_everything(12) == 15


def test_save_rank_creates_parent_directory(tmp_path: Path) -> None:
    output = tmp_path / "logs" / "train.log"
    save_rank("hello", output)
    assert output.read_text(encoding="utf-8") == "hello\n"
