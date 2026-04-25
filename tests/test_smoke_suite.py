"""Tests for the 30-prompt stability suite and its coherence heuristic.

Pure-logic tests — no model load, no GPU, always runnable locally.
"""

from __future__ import annotations

from persona_rag.evaluation.smoke_suite import SUITE, looks_coherent


def test_suite_has_30_prompts():
    assert len(SUITE) == 30


def test_suite_has_10_per_bucket():
    from collections import Counter

    counts = Counter(p.bucket for p in SUITE)
    assert counts == {"factual": 10, "instruction": 10, "persona": 10}


def test_suite_prompt_ids_unique():
    ids = [p.prompt_id for p in SUITE]
    assert len(set(ids)) == len(ids)


def test_persona_bucket_prompts_all_have_system():
    for p in SUITE:
        if p.bucket == "persona":
            assert p.system, f"persona prompt {p.prompt_id} missing system instruction"


def test_non_persona_buckets_have_no_system():
    for p in SUITE:
        if p.bucket != "persona":
            assert p.system is None, (
                f"{p.bucket} prompt {p.prompt_id} should not have a system instruction"
            )


# --------------------------------------------------------------------------- coherence


def test_coherent_output_passes():
    ok, reason = looks_coherent(
        "The capital of Norway is Oslo. It is located on the southern coast of the country."
    )
    assert ok, reason


def test_empty_output_fails():
    ok, _ = looks_coherent("")
    assert not ok


def test_whitespace_only_fails():
    ok, _ = looks_coherent("     \n\n   ")
    assert not ok


def test_short_output_fails():
    ok, reason = looks_coherent("Hi.")
    assert not ok
    assert "too short" in reason


def test_all_punctuation_fails():
    ok, _ = looks_coherent("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    assert not ok


def test_catastrophic_repetition_fails():
    # A severe repeat loop that fp16 quantization instability has been known to produce.
    repeated = ("the quick brown " * 40).strip()
    ok, reason = looks_coherent(repeated)
    assert not ok, f"expected repeat-loop to fail; got ok={ok}, reason={reason}"
    assert "3-gram" in reason or "repeat" in reason.lower(), f"unexpected reason: {reason}"


def test_mildly_repetitive_but_coherent_passes():
    # Real outputs occasionally repeat a phrase; shouldn't trigger the heuristic.
    text = (
        "Exercise is important for cardiovascular health. Regular exercise strengthens "
        "the heart muscle and improves circulation. Exercise also lowers blood pressure "
        "over time, which benefits overall health."
    )
    ok, reason = looks_coherent(text)
    assert ok, reason
