from pathlib import Path

from cypher_extract.paths import DATA_ROOT_ENV, DEFAULT_DATA_ROOT, get_data_root


def test_default_data_root_points_to_download_destination(monkeypatch) -> None:
    monkeypatch.delenv(DATA_ROOT_ENV, raising=False)

    assert DEFAULT_DATA_ROOT == Path(
        "/mnt/local/aiskylimit_new_nothing/cypher-extract/datasets/cypher-extract-data"
    )
    assert get_data_root() == DEFAULT_DATA_ROOT


def test_data_root_can_be_overridden(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv(DATA_ROOT_ENV, str(tmp_path))

    assert get_data_root() == tmp_path
