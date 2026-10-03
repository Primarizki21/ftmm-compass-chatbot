import pytest
from src.knowledge.bm25 import BM25Okapi
from src.knowledge.hybrid_retriever import HybridCourseRetriever


def test_bm25_exact_match():
    corpus = [
        "Mata kuliah MAS2106 Probabilitas dan Teori Peluang",
        "Mata kuliah SIA104 Algoritma dan Pemrograman Dasar",
        "Mata kuliah TIN201 Tata Tulis Ilmiah dan Penulisan Proposal"
    ]
    bm25 = BM25Okapi(corpus)
    res = bm25.search("MAS2106", top_k=1)
    assert len(res) == 1
    assert res[0][0] == 0
    assert res[0][1] > 0.0


def test_bm25_empty_query():
    corpus = ["Dokumen contoh A", "Dokumen contoh B"]
    bm25 = BM25Okapi(corpus)
    res = bm25.search("", top_k=2)
    assert res == []


def test_hybrid_retriever_initialization():
    retriever = HybridCourseRetriever()
    assert len(retriever.courses) > 0
    assert len(retriever.chunks) > 0


def test_hybrid_rrf_structure():
    retriever = HybridCourseRetriever()
    hits, latency = retriever.retrieve_hybrid_rrf("algoritma pemrograman", top_k=3)
    assert len(hits) == 3
    assert latency > 0.0
    for h in hits:
        assert "course_id" in h
        assert "kode_mk" in h
        assert "nama_mk" in h
        assert "rrf_score" in h
        assert h["rrf_score"] > 0.0
