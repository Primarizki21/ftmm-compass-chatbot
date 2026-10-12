import pytest
import numpy as np
from src.guardrails.tier2_semantic_gate import Tier2SemanticGate, GateResult


@pytest.fixture(scope="module")
def gate():
    g = Tier2SemanticGate()
    g.initialize()
    return g


def test_tier2_initialization(gate: Tier2SemanticGate):
    assert gate._initialized is True
    assert gate.academic_vectors is not None
    assert gate.ood_vectors is not None
    assert gate.academic_vectors.shape[0] >= 10
    assert gate.ood_vectors.shape[0] >= 10
    # Check L2 normalization
    ac_norms = np.linalg.norm(gate.academic_vectors, axis=1)
    ood_norms = np.linalg.norm(gate.ood_vectors, axis=1)
    assert np.allclose(ac_norms, 1.0, atol=1e-3)
    assert np.allclose(ood_norms, 1.0, atol=1e-3)


@pytest.mark.parametrize(
    "query",
    [
        "buatkan kode bubble sort",
        "tulis kode python bubble sort",
        "bikinin algoritma sorting bubble sort di c++",
        "bantu bikin game ular pake python",
        "buatkan script web scraping javascript",
    ],
)
def test_coding_ood_queries_blocked(gate: Tier2SemanticGate, query: str):
    res = gate.check(query)
    assert isinstance(res, GateResult)
    assert res.is_ood is True, f"Query '{query}' should be blocked as OOD, got: {res}"
    assert res.reason is not None


@pytest.mark.parametrize(
    "query",
    [
        "bagaimana cara membuat rendang daging sapi yang empuk",
        "berapa prediksi harga bitcoin dan ethereum di tahun 2026",
        "siapa saja calon presiden pemilu amerika serikat",
        "rekomendasi tiket hotel liburan raja ampat",
        "tips menang judi taruhan bola online casino",
        "tulis puisi cinta romantis untuk pacar ulang tahun",
    ],
)
def test_general_lifestyle_ood_blocked(gate: Tier2SemanticGate, query: str):
    res = gate.check(query)
    assert res.is_ood is True, f"Query '{query}' should be blocked as OOD, got: {res}"


@pytest.mark.parametrize(
    "query",
    [
        "Berapa batas maksimal SKS yang boleh saya ambil jika IP semester lalu adalah 3.65?",
        "Apakah saya boleh mengambil mata kuliah Data Mining I jika belum lulus Algoritma Pemrograman I?",
        "Apa saja mata kuliah prasyarat sebelum mengambil Pembelajaran Mesin di Teknik Robotika?",
        "Tolong buatkan rencana studi semester 5 untuk mahasiswa Teknik Industri peminatan Rantai Pasok.",
        "Mata kuliah apa saja yang ditawarkan pada semester 3 Program Studi Rekayasa Nanoteknologi?",
        "Syarat pendaftaran skripsi dan PKL di FTMM apa saja?",
    ],
)
def test_academic_benign_allowed(gate: Tier2SemanticGate, query: str):
    res = gate.check(query)
    assert res.is_ood is False, f"Formal academic query '{query}' must NOT be blocked, got: {res}"


@pytest.mark.parametrize(
    "query",
    [
        "alpro 1 sama alpro 2 bedanya apa bro",
        "Kalo IPS anjlok 2.3 dapet jatah sks berapaan nih bro?",
        "Wajib ngulang ga kalo matkul jarkom dapet nilai D?",
        "Prak terintegrasi itu berat ga sih sizenya berapa sks?",
        "visdat itu belajar apa aja sih kak",
    ],
)
def test_academic_slang_allowed(gate: Tier2SemanticGate, query: str):
    res = gate.check(query)
    assert res.is_ood is False, f"Academic slang query '{query}' must NOT be blocked, got: {res}"


@pytest.mark.parametrize(
    "query",
    [
        "halo selamat pagi kak",
        "hai min selamat siang",
        "terima kasih banyak infonya ya",
    ],
)
def test_greetings_allowed(gate: Tier2SemanticGate, query: str):
    res = gate.check(query)
    assert res.is_ood is False, f"Greeting query '{query}' must NOT be blocked, got: {res}"


def test_gate_latency_efficiency(gate: Tier2SemanticGate):
    res = gate.check("aturan batas pengambilan sks semester depan")
    # Gate latency should be low (<100ms on first run, <50ms typical)
    assert res.latency_ms < 150.0
