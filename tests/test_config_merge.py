from pathlib import Path

from energis.config.merge import PROJECT_ROOT, load_and_merge


def test_load_and_merge_from_subdirectory(monkeypatch):
    repo_root = PROJECT_ROOT
    notebooks_dir = repo_root / "notebooks"
    assert notebooks_dir.is_dir(), "Expected notebooks directory to exist"

    monkeypatch.chdir(notebooks_dir)

    cfg = load_and_merge(["configs/base.yaml"])

    merged_paths = [Path(p).resolve() for p in cfg["meta"]["merged_from"]]
    expected = (repo_root / "configs/base.yaml").resolve()

    assert expected in merged_paths
