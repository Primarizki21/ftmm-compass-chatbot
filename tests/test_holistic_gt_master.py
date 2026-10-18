"""
tests/test_holistic_gt_master.py

Unit and regression tests for holistic ground truth evaluation (Exp-025).
"""

import json
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

from scripts.evaluate_holistic_gt_master import (
    compute_fact_recall,
    compute_ir_metrics,
    compute_cosine_similarity,
    map_expected_route,
    evaluate_dataset,
    DEFAULT_GT_PATH,
)


def test_fact_recall_exact_match():
    gt = "Berdasarkan silabus, mata kuliah Agama Islam 1 (AGI101) berbobot 2 SKS di semester 1."
    ans = "Mata kuliah Agama Islam 1 (AGI101) memiliki bobot 2 SKS pada semester 1."
    recall = compute_fact_recall(ans, gt, source_id="AGI101")
    assert recall >= 0.85


def test_fact_recall_missing_entity():
    gt = "Syarat PKL adalah minimal 80 SKS kumulatif dan telah lulus mata kuliah prasyarat."
    ans = "Halo, selamat datang di FTMM COMPASS."
    recall = compute_fact_recall(ans, gt, source_id="PKL")
    assert recall < 0.20


def test_compute_ir_metrics():
    retrieved = [
        {"course": {"course_id": "TSD101", "kode_mk": "TSD101", "nama_mk": "Kalkulus 1"}, "chunk": ""},
        {"course": {"course_id": "AGI101", "kode_mk": "AGI101", "nama_mk": "Agama Islam 1"}, "chunk": ""},
        {"course": {"course_id": "RN101", "kode_mk": "RN101", "nama_mk": "Pengantar Nanoteknologi"}, "chunk": ""},
    ]
    metrics = compute_ir_metrics(retrieved, target_source_id="AGI101")
    assert metrics["hit@1"] == 0.0
    assert metrics["hit@3"] == 1.0
    assert metrics["hit@5"] == 1.0
    assert metrics["mrr"] == pytest.approx(0.5, 0.01)
    assert metrics["ndcg@5"] > 0.0


def test_map_expected_route():
    assert map_expected_route("Regulasi Akademik") == "REGULATION_SOP"
    assert map_expected_route("SOP Layanan") == "REGULATION_SOP"
    assert map_expected_route("Silabus Wajib") == "COURSE_DISCOVERY"
    assert map_expected_route("Silabus Pilihan") == "COURSE_DISCOVERY"


def test_compute_cosine_similarity():
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([1.0, 0.0, 0.0])
    v3 = np.array([0.0, 1.0, 0.0])
    assert compute_cosine_similarity(v1, v2) == pytest.approx(1.0, 1e-4)
    assert compute_cosine_similarity(v1, v3) == pytest.approx(0.0, 1e-4)


def test_gt_master_file_structure():
    assert DEFAULT_GT_PATH.exists()
    with open(DEFAULT_GT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert len(data) >= 440
    sample = data[0]
    required_keys = {"id", "question", "answer", "source_id", "source_type"}
    assert required_keys.issubset(sample.keys())


def test_clean_thinking_content():
    from src.engine.base import clean_thinking_content

    # 1. Closed <think> tag
    text1 = "<think>Penalaran internal model...</think>Batas maksimal SKS untuk IPS > 3.00 adalah 24 SKS."
    assert clean_thinking_content(text1) == "Batas maksimal SKS untuk IPS > 3.00 adalah 24 SKS."

    # 2. Unclosed <think> tag with trailing content cut off
    text2 = "Jawaban resmi akademik.<think>Penalaran yang terpotong tanpa penutup"
    assert clean_thinking_content(text2) == "Jawaban resmi akademik."

    # 3. Fallback when text only contains thinking block
    text3 = "<think>Konten penalaran yang diekstraksi tanpa tag"
    assert clean_thinking_content(text3) == "Konten penalaran yang diekstraksi tanpa tag"

    # 4. Clean text without think tags
    text4 = "Syarat PKL minimal 80 SKS kumulatif."
    assert clean_thinking_content(text4) == text4


def test_decider_regulation_accuracy_above_90():
    from src.engine.decider import System1Decider, DecisionCategory

    decider = System1Decider()
    decider.initialize()

    test_queries = [
        "kapan kalender akademik semester baru dimulai dan jadwal libur perkuliahan",
        "saya kemarin telat entry krs di cybercampus bagaimana solusinya",
        "prosedur pengisian krs online dan batas waktu perubahan rencana studi",
        "tabel konversi nilai mutu huruf a sampai e dan perhitungan ips ipk",
        "berapa batas maksimal masa studi s1 14 semester sebelum evaluasi do",
        "apa saja kriteria evaluasi putus studi do untuk mahasiswa tahap sarjana",
        "bagaimana regulasi konversi 20 sks untuk program mbkm studi independen",
        "pedoman keikutsertaan program magang studi independen bersertifikat msib mbkm",
        "berapa syarat minimal kehadiran kuliah 75 persen untuk ujian uts dan uas",
        "kebijakan presensi 75 persen dan dispensasi ketidakhadiran sakit izin",
        "apa syarat mengambil capstone project tugas akhir multidisiplin",
        "aturan seminar hasil dan pameran poster capstone project ftmm",
        "bagaimana prosedur pengajuan surat permohonan cuti akademik kuliah di unair",
        "tata tertib etika perkuliahan busana mahasiswa dan sanksi pelanggaran fakultas",
        "mekanisme pengajuan banding nilai sanggah nilai ujian akhir semester kepada dosen",
        "prosedur komplain dan perbaikan nilai khs jika terdapat kekeliruan penilaian dosen",
        "apa saja syarat pkl magang kerja praktek ftmm unair minimal berapa sks",
        "prosedur dan syarat pengajuan proposal skripsi seminar proposal dan tugas akhir",
        "ketentuan yudisium kelulusan wisuda dan bebas tanggungan fakultas",
        "informasi pembayaran ukt keringanan ukt dan banding ukt unair",
    ]

    correct = sum(
        1 for q in test_queries
        if decider.decide(q).category == DecisionCategory.REGULATION_SOP
    )
    accuracy = correct / len(test_queries)
    assert accuracy >= 0.90, f"Akurasi decider regulasi {accuracy*100:.1f}% di bawah target 90%"


def test_pure_cag_engine_execution():
    from src.engine.pure_cag import PureCAGEngine

    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.message.content = "<think>Menganalisis regulasi...</think>Batas maksimal SKS untuk IPS 3.5 adalah 24 SKS."
    mock_resp.get.return_value = 512
    mock_client.chat.return_value = mock_resp

    engine = PureCAGEngine(llm_client=mock_client)
    res = engine.generate_answer("Berapa batas SKS jika IPS 3.5?")

    assert res.architecture == "Pure CAG"
    assert res.answer == "Batas maksimal SKS untuk IPS 3.5 adalah 24 SKS."
    assert "<think>" not in res.answer
    assert res.tokens_generated == 512
    assert res.retrieval_latency_ms == 0.0

    # Assert num_predict passed as 512
    _, kwargs = mock_client.chat.call_args
    assert kwargs.get("options", {}).get("num_predict") == 512


def test_pure_rag_engine_execution():
    from src.engine.pure_rag import PureRAGEngine

    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.message.content = "<think>Mencari info silabus...</think>Mata kuliah Kalkulus 1 (TSD101) berbobot 3 SKS."
    mock_resp.get.return_value = 512
    mock_client.chat.return_value = mock_resp

    engine = PureRAGEngine(llm_client=mock_client, top_k=2)
    res = engine.generate_answer("Informasi Kalkulus 1")

    assert res.architecture == "Pure RAG"
    assert res.answer == "Mata kuliah Kalkulus 1 (TSD101) berbobot 3 SKS."
    assert "<think>" not in res.answer
    assert res.tokens_generated == 512
    assert res.retrieval_latency_ms >= 0.0

    # Assert num_predict passed as 512
    _, kwargs = mock_client.chat.call_args
    assert kwargs.get("options", {}).get("num_predict") == 512
