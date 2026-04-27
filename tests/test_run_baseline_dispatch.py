"""Tests for the script-level helpers in `scripts/run_baseline.py`.

The hydra entry-point itself is exercised by an end-to-end run on the V100;
here we cover the pure-logic dispatch pieces (schema parsing, seed selection,
filename composition) so a config-shape regression fails fast in local CI.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_baseline.py"


def _load_run_baseline_module():
    """Import `scripts/run_baseline.py` as a module without invoking Hydra."""
    spec = importlib.util.spec_from_file_location("_test_run_baseline", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def rb():
    return _load_run_baseline_module()


# ----------------------------------------------------------- _parse_test_queries


def test_parse_accepts_minimal_query(rb) -> None:
    parsed = rb._parse_test_queries([{"text": "hello", "bucket": "knowledge_grounded"}])
    assert len(parsed) == 1
    q = parsed[0]
    assert q.text == "hello"
    assert q.bucket == "knowledge_grounded"
    assert q.multi_seed is False
    assert q.skip_in_matrix is False
    assert q.ix == 0


def test_parse_accepts_omegaconf_dictconfig(rb) -> None:
    """The Hydra resolver hands us DictConfig items, not plain dicts."""
    raw = OmegaConf.create(
        [
            {"text": "alpha", "bucket": "knowledge_grounded"},
            {
                "text": "beta",
                "bucket": "constraint_stressing",
                "multi_seed": True,
                "skip_in_matrix": False,
            },
        ]
    )
    parsed = rb._parse_test_queries(raw)
    assert [q.text for q in parsed] == ["alpha", "beta"]
    assert parsed[1].multi_seed is True


def test_parse_rejects_unknown_bucket(rb) -> None:
    with pytest.raises(ValueError, match="bucket"):
        rb._parse_test_queries([{"text": "x", "bucket": "nonsense"}])


def test_parse_rejects_missing_text(rb) -> None:
    with pytest.raises(ValueError, match="text"):
        rb._parse_test_queries([{"bucket": "knowledge_grounded"}])


def test_parse_rejects_string_for_top_level(rb) -> None:
    with pytest.raises(ValueError, match="list of mappings"):
        rb._parse_test_queries("not a list")


def test_parse_rejects_legacy_flat_string_list(rb) -> None:
    """The old `list[str]` format is intentionally a hard cut — fail loudly."""
    with pytest.raises(ValueError, match="must be a mapping"):
        rb._parse_test_queries(["plain string query"])


def test_parse_rejects_empty_list(rb) -> None:
    with pytest.raises(ValueError, match="empty"):
        rb._parse_test_queries([])


def test_parse_rejects_none(rb) -> None:
    with pytest.raises(ValueError, match="missing"):
        rb._parse_test_queries(None)


# ----------------------------------------------------------- _seeds_for


def test_seeds_for_single_query_uses_default(rb) -> None:
    q = rb._ParsedQuery(
        ix=0, text="x", bucket="knowledge_grounded", multi_seed=False, skip_in_matrix=False
    )
    assert rb._seeds_for(q, default_seed=42, multi_seeds=[1, 2, 3]) == [42]


def test_seeds_for_multi_query_uses_full_list(rb) -> None:
    q = rb._ParsedQuery(
        ix=1, text="y", bucket="constraint_stressing", multi_seed=True, skip_in_matrix=False
    )
    assert rb._seeds_for(q, default_seed=42, multi_seeds=[1, 2, 3]) == [1, 2, 3]


def test_seeds_for_multi_query_with_empty_seed_list_raises(rb) -> None:
    q = rb._ParsedQuery(
        ix=2, text="z", bucket="constraint_stressing", multi_seed=True, skip_in_matrix=False
    )
    with pytest.raises(ValueError, match="constraint_query_seeds"):
        rb._seeds_for(q, default_seed=42, multi_seeds=[])


# ----------------------------------------------------------- _response_filename


def test_response_filename_single_seed_omits_seed(rb) -> None:
    q = rb._ParsedQuery(
        ix=3, text="x", bucket="knowledge_grounded", multi_seed=False, skip_in_matrix=False
    )
    assert rb._response_filename(q, seed=42) == "response_03.json"


def test_response_filename_multi_seed_includes_seed(rb) -> None:
    q = rb._ParsedQuery(
        ix=7, text="x", bucket="constraint_stressing", multi_seed=True, skip_in_matrix=False
    )
    assert rb._response_filename(q, seed=1337) == "response_07_seed1337.json"
    assert rb._response_filename(q, seed=42) == "response_07_seed0042.json"


# ----------------------------------------------------------- shipped baseline.yaml


def test_shipped_baseline_yaml_parses_and_buckets_balance(rb) -> None:
    """The shipped `baseline.yaml` must parse cleanly under the new schema.

    Also asserts the bucket distribution matches the pilot intent: at least
    one knowledge_grounded, semantic_adjacent, constraint_stressing query,
    plus the contamination-demo that's kept on disk but skipped from the
    matrix.
    """
    cfg_path = REPO_ROOT / "src" / "persona_rag" / "config" / "baseline.yaml"
    cfg = OmegaConf.load(cfg_path)
    parsed = rb._parse_test_queries(cfg.test_queries)
    by_bucket: dict[str, int] = {}
    for q in parsed:
        by_bucket[q.bucket] = by_bucket.get(q.bucket, 0) + 1
    assert by_bucket.get("knowledge_grounded", 0) >= 1
    assert by_bucket.get("semantic_adjacent", 0) >= 1
    assert by_bucket.get("constraint_stressing", 0) >= 1
    assert by_bucket.get("appendix_contamination_demo", 0) >= 1
    # The contamination demo should be flagged out of the headline matrix.
    contam = [q for q in parsed if q.bucket == "appendix_contamination_demo"]
    assert all(q.skip_in_matrix for q in contam)
    # All constraint_stressing queries should run multi-seed by design.
    constraint = [q for q in parsed if q.bucket == "constraint_stressing"]
    assert all(q.multi_seed for q in constraint)
