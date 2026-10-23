import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from src.engine.decider import DecisionCategory, DecisionResult, System1Decider
from src.knowledge.hyde_generator import HydeGenerator
from src.knowledge.retriever import CourseRetriever


def test_hyde_generator_prompt_building():
    generator = HydeGenerator(model_name="llama3.2:3b", max_tokens=100, temperature=0.3)
    prompt = generator.build_prompt("Gimana cara ngoding drone?", prodi_hint="TRKB")
    assert "FTMM UNAIR" in prompt
    assert "TRKB" in prompt
    assert "Gimana cara ngoding drone?" in prompt
    assert "Draf Silabus Hipotesis:" in prompt


def test_hyde_generator_fallback_on_error():
    generator = HydeGenerator()
    mock_client = MagicMock()
    mock_client.generate.side_effect = RuntimeError("Ollama connection refused")
    generator._client = mock_client

    result = generator.generate("Kueri uji fallback")
    assert result == "Kueri uji fallback"


def test_course_retriever_retrieve_by_vector():
    retriever = CourseRetriever()
    retriever.load()
    assert retriever._loaded is True
    assert retriever.index is not None

    # Dummy random unit vector with matching dimension
    dim = retriever.dimension or 768
    dummy_vec = np.random.randn(dim).astype(np.float32)
    dummy_vec /= np.linalg.norm(dummy_vec)

    results = retriever.retrieve_by_vector(dummy_vec, top_k=3)
    assert len(results) == 3
    assert "score" in results[0]
    assert "course" in results[0]
    assert "chunk" in results[0]


def test_course_retriever_retrieve_with_hyde_mocked():
    retriever = CourseRetriever()
    retriever.load()

    mock_generator = MagicMock()
    mock_generator.generate.return_value = "Estimasi Nama Mata Kuliah: Mobile Robot\nTopik: Navigasi Otonom"

    results, hypo_text = retriever.retrieve_with_hyde(
        query="robot yang bisa jalan sendiri di trkb apa ya",
        top_k=2,
        prodi_filter="Teknik Robotika",
        generator=mock_generator,
        return_hypo=True,
    )
    assert len(results) <= 2
    assert "Mobile Robot" in hypo_text
    assert mock_generator.generate.called


def test_decider_should_use_hyde_rules():
    decider = System1Decider()
    decider.initialize()

    # 1. Chat only -> False
    chat_dec = DecisionResult(
        category=DecisionCategory.CHAT_ONLY,
        confidence=0.85,
        selected_tools=None,
        latency_ms=1.0,
        embedding_latency_ms=0.5,
        scores={},
        margin=0.20,
    )
    assert decider.should_use_hyde("Halo selamat pagi", decision=chat_dec) is False

    # 2. Regulation SOP -> False
    sop_dec = DecisionResult(
        category=DecisionCategory.REGULATION_SOP,
        confidence=0.80,
        selected_tools=None,
        latency_ms=1.0,
        embedding_latency_ms=0.5,
        scores={},
        margin=0.15,
    )
    assert decider.should_use_hyde("Berapa batas SKS semester 3?", decision=sop_dec) is False

    # 3. Explicit Course Code in query -> False
    code_dec = DecisionResult(
        category=DecisionCategory.COURSE_DISCOVERY,
        confidence=0.50,
        selected_tools=None,
        latency_ms=1.0,
        embedding_latency_ms=0.5,
        scores={},
        margin=0.01,
    )
    assert decider.should_use_hyde("Apa silabus mata kuliah SID206 di TSD?", decision=code_dec) is False

    # 4. Slang/Ambiguous without explicit code -> True
    ambiguous_dec = DecisionResult(
        category=DecisionCategory.COURSE_DISCOVERY,
        confidence=0.48,
        selected_tools=None,
        latency_ms=1.0,
        embedding_latency_ms=0.5,
        scores={},
        margin=0.015,
    )
    assert decider.should_use_hyde("matkul ngoding drone otonom yang seru apa ya", decision=ambiguous_dec) is True
