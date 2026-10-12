import pytest
from src.engine.decider import get_shared_gemma2_model
from src.engine.react_agent import get_system1_decider
from src.knowledge.retriever import CourseRetriever, get_course_retriever
from src.knowledge.hybrid_retriever import HybridCourseRetriever

def test_shared_singleton_instance():
    """Verify Single-Model Residency: Decider and Retriever MUST share the exact same model object."""
    decider = get_system1_decider()
    retriever = CourseRetriever(backend="gemma2", dimension=256)
    
    decider_model = decider._get_gemma_model()
    retriever_model = retriever._get_gemma_model()
    shared_model = get_shared_gemma2_model()

    assert decider_model is not None
    assert retriever_model is not None
    assert id(decider_model) == id(retriever_model)
    assert id(retriever_model) == id(shared_model)


def test_gemma2_course_retrieval_256d():
    """Verify dense retrieval with MRL 256d index."""
    retriever = CourseRetriever(backend="gemma2", dimension=256)
    results = retriever.retrieve("algoritma pemrograman python", top_k=3)
    assert len(results) == 3
    assert any("algoritma" in r["course"]["nama_mk"].lower() or "pemrograman" in r["course"]["nama_mk"].lower() for r in results)
    assert results[0]["score"] > 0.0


def test_gemma2_course_retrieval_768d():
    """Verify dense retrieval with full 768d index."""
    retriever = CourseRetriever(backend="gemma2", dimension=768)
    results = retriever.retrieve("kalkulus differensial integral", top_k=3)
    assert len(results) == 3
    assert any("kalkulus" in r["course"]["nama_mk"].lower() for r in results)


def test_gemma2_hybrid_retriever():
    """Verify Hybrid RRF works with Gemma-2 256d dense backbone."""
    dense_retriever = CourseRetriever(backend="gemma2", dimension=256)
    hybrid_retriever = HybridCourseRetriever(dense_retriever=dense_retriever)
    
    hits, latency = hybrid_retriever.retrieve_hybrid_rrf("jaringan komputer", top_k=3)
    assert len(hits) == 3
    assert latency > 0.0
    assert any("jaringan" in h["nama_mk"].lower() or "komputer" in h["nama_mk"].lower() for h in hits)
