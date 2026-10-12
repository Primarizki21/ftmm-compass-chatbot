import numpy as np
import pytest

from src.engine.decider import (
    DecisionCategory,
    System1Decider,
    CENTROIDS_GEMMA2_PATH,
)


def test_gemma2_initialization():
    decider = System1Decider(backend="gemma2")
    decider.initialize()
    assert decider.centroid_matrix is not None
    assert decider.centroid_matrix.shape == (5, 768)
    norms = np.linalg.norm(decider.centroid_matrix, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_gemma2_mrl_truncation_shapes():
    decider = System1Decider(backend="gemma2")
    decider.initialize()

    for dim in [512, 256, 128]:
        decider.set_mrl_dim(dim)
        assert decider.centroid_matrix.shape == (5, dim)
        norms = np.linalg.norm(decider.centroid_matrix, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    # Reset back to full 768d
    decider.set_mrl_dim(None)
    assert decider.centroid_matrix.shape == (5, 768)


def test_gemma2_fast_path_regex():
    decider = System1Decider(backend="gemma2")
    res = decider.decide("Daftar mata kuliah semester 3")
    assert res.category == DecisionCategory.COURSE_DISCOVERY
    assert res.confidence == 1.0
    assert res.margin == 1.0
    assert res.selected_tools is not None


def test_gemma2_mock_embedding_decide():
    # Test end-to-end decision routing without loading 1.4GB model in unit test
    decider = System1Decider(backend="gemma2")
    decider.initialize()

    # Create mock embedding identical to CHAT_ONLY centroid + slight noise
    chat_centroid = decider.centroid_matrix[0]
    noisy_vec = chat_centroid + np.random.normal(0, 0.01, size=chat_centroid.shape).astype(np.float32)
    noisy_vec = noisy_vec / np.linalg.norm(noisy_vec)

    class MockModel:
        def encode(self, text, **kwargs):
            return noisy_vec

    decider._gemma_model = MockModel()
    res = decider.decide("Halo selamat pagi bro!")

    assert res.category == DecisionCategory.CHAT_ONLY
    assert res.confidence > 0.80
    assert res.margin > 0.0
    assert res.selected_tools is None


def test_gemma2_mrl_decide_with_mock():
    decider = System1Decider(backend="gemma2")
    decider.initialize()

    # Create mock embedding matching SKS_CALCULATION centroid
    sks_centroid = decider.centroid_matrix[2]

    class MockModel:
        def encode(self, text, **kwargs):
            return sks_centroid

    decider._gemma_model = MockModel()

    for dim in [768, 512, 256, 128]:
        decider.set_mrl_dim(dim)
        res = decider.decide("Berapa batas SKS jika IPS 3.5?")
        assert res.category == DecisionCategory.SKS_CALCULATION
        assert res.confidence > 0.95
        assert res.margin > 0.0
        assert res.selected_tools is not None
        assert any(t["function"]["name"] == "calculate_sks_limit" for t in res.selected_tools)
