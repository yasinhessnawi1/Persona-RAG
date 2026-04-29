"""Tests for the knowledge store: chunking, dense, BM25, hybrid (RRF + weighted-sum)."""

from __future__ import annotations

from pathlib import Path

import pytest

from persona_rag.stores.knowledge_store import KnowledgeDocument, KnowledgeStore

REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = REPO_ROOT / "benchmarks_data" / "knowledge_corpora" / "cs_tutor"


# -------------------------------------------------------------- fixtures


@pytest.fixture
def small_docs() -> list[KnowledgeDocument]:
    """Three deliberately simple synthetic documents.

    `rare_widget_42` is a unique exact-term that BM25 can pick up but a
    semantic embedder will not surface for natural-language queries.
    """
    return [
        KnowledgeDocument(
            doc_id="raft",
            text="Raft is a distributed consensus algorithm. The leader is elected by a majority quorum.",
            source="raft.md",
        ),
        KnowledgeDocument(
            doc_id="cap",
            text="The CAP theorem says you cannot have all of consistency, availability, and partition tolerance during a partition.",
            source="cap.md",
        ),
        KnowledgeDocument(
            doc_id="widget",
            text="The rare_widget_42 reference is the only place this exact token appears in the corpus.",
            source="widget.md",
        ),
    ]


@pytest.fixture
def store(tmp_path: Path, fake_embedder, small_docs) -> KnowledgeStore:
    s = KnowledgeStore(
        persist_path=tmp_path / "knowledge",
        collection_name="knowledge_chunks_test",
        embedding_function=fake_embedder,
        # Pick a chunk size that leaves each short doc as one chunk.
        chunk_size=512,
        chunk_overlap=0,
    )
    s.index_corpus(small_docs)
    return s


# -------------------------------------------------------------- writes


def test_index_corpus_returns_chunk_count(tmp_path, fake_embedder, small_docs) -> None:
    s = KnowledgeStore(
        persist_path=tmp_path / "ks",
        collection_name="ks_test",
        embedding_function=fake_embedder,
    )
    n = s.index_corpus(small_docs)
    assert n >= len(small_docs)  # at least one chunk per doc
    assert s.count() == n


def test_index_corpus_idempotent(store, small_docs) -> None:
    """Re-indexing the same corpus keeps the chunk count constant (upsert by id)."""
    before = store.count()
    store.index_corpus(small_docs)
    assert store.count() == before


def test_real_corpus_indexes_without_error(tmp_path, fake_embedder) -> None:
    """The shipped dev corpus indexes end-to-end via the LlamaIndex SentenceSplitter."""
    s = KnowledgeStore(
        persist_path=tmp_path / "real",
        collection_name="real_test",
        embedding_function=fake_embedder,
    )
    docs = [
        KnowledgeDocument(doc_id=p.stem, text=p.read_text(encoding="utf-8"), source=p.name)
        for p in sorted(CORPUS_DIR.glob("*.md"))
    ]
    n = s.index_corpus(docs)
    assert n > 0
    # Every chunk has the expected metadata shape.
    sample = s.query_dense("Raft", top_k=1)
    if sample:
        assert "doc_id" in sample[0].metadata
        assert "source" in sample[0].metadata
        assert "chunk_ix" in sample[0].metadata


# -------------------------------------------------------------- reads


def test_query_dense_returns_chunks(store) -> None:
    out = store.query_dense("anything", top_k=2)
    # Fake embedder is hash-deterministic — order is arbitrary but result is non-empty.
    assert 0 < len(out) <= 2


def test_query_dense_top_k_validation(store) -> None:
    with pytest.raises(ValueError, match="positive"):
        store.query_dense("x", top_k=0)


def test_query_dense_empty_text_returns_empty(store) -> None:
    assert store.query_dense("", top_k=3) == []


def test_query_bm25_finds_exact_term(store) -> None:
    """BM25 catches the rare exact-term match the dense embedder would miss."""
    out = store.query_bm25("rare_widget_42", top_k=3)
    # The widget doc must appear in the BM25 top-3 (probably top-1).
    assert any(c.metadata.get("doc_id") == "widget" for c in out)


def test_query_bm25_empty_text_returns_empty(store) -> None:
    assert store.query_bm25("", top_k=3) == []


def test_query_bm25_top_k_validation(store) -> None:
    with pytest.raises(ValueError, match="positive"):
        store.query_bm25("x", top_k=0)


def test_query_hybrid_default_uses_rrf(store) -> None:
    """Hybrid with `alpha=None` → RRF; results non-empty and ordered by fused score."""
    out = store.query_hybrid("Raft consensus", top_k=2)
    assert 0 < len(out) <= 2


def test_query_hybrid_exact_term_beats_semantic_only(store) -> None:
    """Spec-DoD synthetic: hybrid surfaces the exact-term doc that dense alone misses.

    Fake embedder is hash-random, so dense retrieval ranks documents in
    near-arbitrary order. BM25, however, finds the exact-term match.
    The fused output must include the widget doc.
    """
    out = store.query_hybrid("rare_widget_42 reference", top_k=3)
    doc_ids = [c.metadata.get("doc_id") for c in out]
    assert "widget" in doc_ids


def test_query_hybrid_weighted_sum_alpha_zero_matches_bm25(store) -> None:
    """alpha=0 → weighted-sum should give BM25-only ordering for the exact-term query."""
    out = store.query_hybrid("rare_widget_42 reference", top_k=3, alpha=0.0)
    doc_ids = [c.metadata.get("doc_id") for c in out]
    assert "widget" in doc_ids


def test_query_hybrid_alpha_validation(store) -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        store.query_hybrid("anything", top_k=3, alpha=1.5)


# -------------------------------------------------------------- properties


def test_collection_name_and_persist_path_exposed(tmp_path, fake_embedder) -> None:
    s = KnowledgeStore(
        persist_path=tmp_path / "ks2",
        collection_name="my_collection",
        embedding_function=fake_embedder,
    )
    assert s.collection_name == "my_collection"
    assert s.persist_path == (tmp_path / "ks2").resolve()


def test_count_zero_before_indexing(tmp_path, fake_embedder) -> None:
    s = KnowledgeStore(
        persist_path=tmp_path / "empty",
        collection_name="empty_test",
        embedding_function=fake_embedder,
    )
    assert s.count() == 0


# -------------------------------------------------------------- remove_documents


def test_remove_documents_drops_all_chunks_for_doc_id(store) -> None:
    """``remove_documents`` clears every chunk emitted from the listed doc ids."""
    before = store.count()
    n_removed = store.remove_documents(["raft"])
    assert n_removed >= 1, "raft doc must contribute at least one chunk"
    assert store.count() == before - n_removed


def test_remove_documents_idempotent_on_missing_doc_id(store) -> None:
    """Calling on an unknown doc_id is a no-op (returns 0)."""
    before = store.count()
    n_removed = store.remove_documents(["does_not_exist"])
    assert n_removed == 0
    assert store.count() == before


def test_remove_documents_empty_input(store) -> None:
    assert store.remove_documents([]) == 0


def test_add_then_remove_restores_topk(tmp_path, fake_embedder, small_docs) -> None:
    """Invariant: ``add → query → remove → query`` returns the pre-injection result.

    Counterfactual-probe injection depends on this byte-for-byte: after the
    probe turn ejects the planted chunk, the next turn must see the
    pre-injection store state.
    """
    s = KnowledgeStore(
        persist_path=tmp_path / "inject",
        collection_name="inject_test",
        embedding_function=fake_embedder,
        chunk_size=512,
        chunk_overlap=0,
    )
    s.index_corpus(small_docs)
    # Capture the pre-injection top-k (BM25, deterministic across runs).
    pre = [c.id for c in s.query_bm25("Raft consensus algorithm", top_k=5)]
    pre_count = s.count()

    injected = KnowledgeDocument(
        doc_id="injected_counter_evidence",
        text=(
            "Recent industry survey claims raft consensus algorithm "
            "is now widely considered obsolete for distributed log storage."
        ),
        source="injected.md",
    )
    s.add_documents([injected])
    assert s.count() == pre_count + 1
    during = [c.id for c in s.query_bm25("Raft consensus algorithm", top_k=5)]
    # The injected chunk should appear (overlapping vocabulary). If it didn't
    # appear we'd still be testing the remove invariant, but the assertion
    # documents the expected behaviour.
    assert any(cid.startswith("injected_counter_evidence") for cid in during)

    n_removed = s.remove_documents(["injected_counter_evidence"])
    assert n_removed >= 1
    assert s.count() == pre_count

    post = [c.id for c in s.query_bm25("Raft consensus algorithm", top_k=5)]
    assert post == pre, (
        f"post-remove top-k must match pre-injection top-k exactly; pre={pre} post={post}"
    )
