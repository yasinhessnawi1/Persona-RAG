"""Tests for the `Response` Pydantic contract — protects mechanism specs from drift."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from persona_rag.retrieval.base import Response, RetrievalPipeline, Turn
from persona_rag.schema.chunker import PersonaChunk
from persona_rag.stores.knowledge_chunk import KnowledgeChunk


def test_response_extra_forbid_rejects_unknown_field() -> None:
    """`extra='forbid'` — typos in mechanism specs fail loudly, not silently."""
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        Response(
            text="hi", prompt_used="p", retrieved_personas={}
        )  # typo: persona*s*  # type: ignore[call-arg]


def test_response_defaults_steering_and_drift_to_none_for_b1_b2() -> None:
    """B1 / B2 don't touch steering or drift — defaults stay False / None."""
    r = Response(text="x", prompt_used="p")
    assert r.steering_applied is False
    assert r.drift_signal is None
    assert r.retrieved_persona == {}
    assert r.retrieved_knowledge == []


def test_response_accepts_typed_persona_dict() -> None:
    """`retrieved_persona` keys must be `ChunkKind` literals."""
    chunk = PersonaChunk(
        id="p:identity:0",
        text="Name: Marcus",
        kind="identity",
        metadata={"persona_id": "p", "kind": "identity"},
    )
    r = Response(text="hi", prompt_used="p", retrieved_persona={"identity": [chunk]})
    assert r.retrieved_persona["identity"][0].id == "p:identity:0"


def test_response_accepts_knowledge_chunks() -> None:
    chunk = KnowledgeChunk(
        id="d:0", text="foo", metadata={"doc_id": "d", "source": "f", "chunk_ix": "0"}
    )
    r = Response(text="hi", prompt_used="p", retrieved_knowledge=[chunk])
    assert r.retrieved_knowledge[0].id == "d:0"


def test_turn_role_validation() -> None:
    """`Turn.role` must be `'user'` or `'assistant'`."""
    Turn(role="user", content="hi")
    Turn(role="assistant", content="hi")
    with pytest.raises(ValidationError):
        Turn(role="system", content="hi")


def test_retrieval_pipeline_protocol_runtime_checkable() -> None:
    """Anything with `name` + `respond` satisfies `RetrievalPipeline` at runtime."""

    class _StubPipe:
        name = "stub"

        def respond(self, query, persona, history=None):
            return Response(text="x", prompt_used="p")

    assert isinstance(_StubPipe(), RetrievalPipeline)
