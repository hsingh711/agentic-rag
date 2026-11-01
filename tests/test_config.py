"""Tests for the config loader."""

import pytest

from src.config import load_config

REQUIRED_SECTIONS = ["ingest", "embedding", "vector_store", "retrieval", "llm", "classifier", "critic", "verifier"]


def test_load_default_config_has_required_sections():
    config = load_config()
    for section in REQUIRED_SECTIONS:
        assert section in config


def test_key_tunables_present():
    config = load_config()
    assert config["ingest"]["chunk_size_tokens"] == 400
    assert config["retrieval"]["top_k"] > 0
    assert config["critic"]["max_retries"] >= 0
    assert config["llm"]["roles"]["generator"] in config["llm"]["models"]


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


def test_non_mapping_raises(tmp_path):
    bad_file = tmp_path / "bad.yaml"
    bad_file.write_text("- just\n- a list\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(bad_file)
