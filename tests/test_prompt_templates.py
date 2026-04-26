"""Tests for prompt templates: vanilla, structured persona block, audit shim, token budget."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from persona_rag.retrieval.prompt_templates import (
    B1_VANILLA_RAG_SYSTEM,
    FewShotBundle,
    estimate_token_count,
    render_b1_user_block,
    render_b2_persona_block,
    render_b2_simple_system,
    render_b2_user_block,
    trim_chunks_to_token_budget,
)
from persona_rag.schema.persona import Persona
from persona_rag.stores.knowledge_chunk import KnowledgeChunk

REPO_ROOT = Path(__file__).resolve().parents[1]
PERSONAS_DIR = REPO_ROOT / "personas"
EXAMPLES_DIR = REPO_ROOT / "personas" / "examples"


# -------------------------------------------------------------- fixtures


@pytest.fixture
def cs_tutor() -> Persona:
    return Persona.from_yaml(PERSONAS_DIR / "cs_tutor.yaml")


@pytest.fixture
def cs_tutor_few_shots() -> FewShotBundle:
    return FewShotBundle.from_yaml(EXAMPLES_DIR / "cs_tutor.yaml")


@pytest.fixture
def small_corpus() -> list[KnowledgeChunk]:
    return [
        KnowledgeChunk(
            id="raft:0",
            text="Raft elects a leader by majority vote.",
            metadata={"source": "raft.md", "doc_id": "raft", "chunk_ix": "0"},
        ),
        KnowledgeChunk(
            id="cap:0",
            text="CAP says you can't have all of C, A, P during partition.",
            metadata={"source": "cap.md", "doc_id": "cap", "chunk_ix": "0"},
        ),
    ]


# -------------------------------------------------------------- B1


def test_b1_vanilla_system_is_neutral_strawman() -> None:
    """B1 system prompt is the literature-standard neutral framing — no persona shape."""
    assert "helpful assistant" in B1_VANILLA_RAG_SYSTEM
    assert "persona" not in B1_VANILLA_RAG_SYSTEM.lower()


def test_b1_user_block_numbers_passages_and_includes_query(small_corpus) -> None:
    block = render_b1_user_block("What is Raft?", small_corpus)
    assert "[1]" in block
    assert "[2]" in block
    assert "What is Raft?" in block
    assert "Raft elects" in block


def test_b1_user_block_handles_empty_retrieval() -> None:
    block = render_b1_user_block("Hello", [])
    assert "Retrieved passages" not in block
    assert "Hello" in block


# -------------------------------------------------------------- structured persona block


def test_b2_persona_block_has_identity_facts_worldview_constraints(
    cs_tutor, cs_tutor_few_shots
) -> None:
    """Structured persona block includes identity, self-facts, worldview, constraints, and few-shots."""
    block = render_b2_persona_block(cs_tutor, cs_tutor_few_shots)
    assert cs_tutor.identity.name in block
    assert cs_tutor.identity.role in block
    assert "About yourself:" in block
    assert "Your views" in block
    assert "You must NOT:" in block
    assert "Example exchanges" in block
    # Every constraint surfaces.
    for c in cs_tutor.identity.constraints:
        assert c in block


def test_b2_persona_block_orders_constraint_case_last(cs_tutor, cs_tutor_few_shots) -> None:
    """Constraint-case few-shot is last so it's freshest in the model's context."""
    block = render_b2_persona_block(cs_tutor, cs_tutor_few_shots)
    constraint_titles = [ex.title for ex in cs_tutor_few_shots.exchanges if ex.is_constraint_case]
    assert constraint_titles, "test fixture must contain at least one constraint case"
    last_constraint = constraint_titles[-1]
    last_pos = block.rfind(last_constraint)
    # Every non-constraint title must appear before the last constraint title.
    for ex in cs_tutor_few_shots.exchanges:
        if not ex.is_constraint_case:
            assert block.find(ex.title) < last_pos


def test_b2_persona_block_emits_epistemic_tag_for_non_fact_worldview(
    cs_tutor, cs_tutor_few_shots
) -> None:
    """Worldview claims tagged `belief`/`contested` show their tag in parens."""
    block = render_b2_persona_block(cs_tutor, cs_tutor_few_shots)
    # cs_tutor has at least one `contested` and several `belief`s.
    assert "(contested)" in block
    assert "(belief)" in block
    # `fact`-tagged worldview claims are not annotated (cs_tutor's CAP/PACELC claim).
    cap_claim = next(w for w in cs_tutor.worldview if w.epistemic == "fact")
    assert cap_claim.claim in block
    assert f"{cap_claim.claim}(fact)" not in block  # no inline tag for facts


def test_b2_persona_block_high_confidence_facts_unannotated(cs_tutor, cs_tutor_few_shots) -> None:
    """Self-facts at confidence >= 0.95 are presented without a confidence tag."""
    block = render_b2_persona_block(cs_tutor, cs_tutor_few_shots)
    high_conf = next(sf for sf in cs_tutor.self_facts if sf.confidence >= 0.99)
    assert f"  - {high_conf.fact}" in block
    assert f"  - {high_conf.fact}  (confidence" not in block


def test_b2_user_block_does_not_include_persona_chunks(cs_tutor, small_corpus) -> None:
    """B2 puts persona in the system block; user block carries only retrieved knowledge."""
    block = render_b2_user_block(
        "What is CAP?", small_corpus, persona_chunks_by_kind={"identity": []}
    )
    assert cs_tutor.identity.name not in block
    assert "[1]" in block
    assert "What is CAP?" in block


# -------------------------------------------------------------- B2 audit shim


def test_b2_simple_system_is_one_liner(cs_tutor) -> None:
    """The audit-only shim is a one-line persona prefix — by construction shorter than the structured block."""
    simple = render_b2_simple_system(cs_tutor)
    assert cs_tutor.identity.name in simple
    assert cs_tutor.identity.role in simple
    # No structured sections.
    assert "About yourself:" not in simple
    assert "You must NOT:" not in simple


# -------------------------------------------------------------- few-shots


def test_few_shot_bundle_loads_for_every_shipped_persona() -> None:
    """All three personas have a few-shot bundle alongside their YAML."""
    for persona_id in ("cs_tutor", "historian", "climate_scientist"):
        bundle = FewShotBundle.from_yaml(EXAMPLES_DIR / f"{persona_id}.yaml")
        assert bundle.persona_id == persona_id
        # Each shipped few-shot bundle has 5 exchanges.
        assert len(bundle.exchanges) == 5
        # Exactly one exchange per persona is the constraint-case demo.
        constraint_count = sum(1 for ex in bundle.exchanges if ex.is_constraint_case)
        assert constraint_count == 1


def test_few_shot_bundle_rejects_unknown_keys(tmp_path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "persona_id: bad\n"
        "exchanges:\n"
        "  - title: x\n"
        "    is_constraint_case: false\n"
        "    turns:\n"
        "      - role: user\n"
        "        content: x\n"
        "      - role: assistant\n"
        "        content: y\n"
        "    spurious: wat\n",
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        FewShotBundle.from_yaml(bad)


def test_few_shot_turn_role_validation(tmp_path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "persona_id: bad\n"
        "exchanges:\n"
        "  - title: x\n"
        "    turns:\n"
        "      - role: system\n"
        "        content: x\n"
        "      - role: assistant\n"
        "        content: y\n",
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        FewShotBundle.from_yaml(bad)


# -------------------------------------------------------------- token budget


def test_token_estimate_is_roughly_chars_over_4() -> None:
    assert estimate_token_count("a" * 8) == 2
    assert estimate_token_count("") == 1  # floor; never zero


def test_trim_chunks_to_token_budget_keeps_top_ranked() -> None:
    # Each chunk: ~200 chars → 50 tokens via the chars/4 estimate, +16 citation
    # overhead = 66 tokens per chunk. 5 chunks → 330 tokens. Budget chosen so
    # only ~3 fit: 800 - 256 - 128 - 200 = 216 tokens → 3 chunks (3*66=198 ≤ 216).
    chunks = [KnowledgeChunk(id=f"d:{i}", text="x" * 200, metadata={}) for i in range(5)]
    kept, dropped = trim_chunks_to_token_budget(
        chunks, fixed_overhead_tokens=200, max_input_tokens=800, max_new_tokens=256
    )
    assert 0 < len(kept) < 5
    assert dropped == 5 - len(kept)
    # The dropped chunks are the lowest-ranked (last in the list).
    assert kept == chunks[: len(kept)]


def test_trim_chunks_drops_all_when_budget_exhausted_by_overhead() -> None:
    chunks = [KnowledgeChunk(id="d:0", text="x" * 100, metadata={})]
    kept, dropped = trim_chunks_to_token_budget(
        chunks, fixed_overhead_tokens=10_000, max_input_tokens=4096, max_new_tokens=256
    )
    assert kept == []
    assert dropped == 1
