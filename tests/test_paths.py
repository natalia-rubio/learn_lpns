"""Tests for repository path helpers."""

from learn_lpns.tools.paths import repo_root


def test_repo_root_contains_config_defaults():
    root = repo_root()
    assert root.is_dir()
    assert (root / "config" / "defaults.yaml").is_file() or (root / "learn_lpns" / "config" / "defaults.yaml").is_file()
