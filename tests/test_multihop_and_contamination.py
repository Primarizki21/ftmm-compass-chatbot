import json
from pathlib import Path

import numpy as np
import pytest

from scripts.evaluate_multihop_and_contamination import (
    compute_cosine_similarity,
    compute_ndcg,
    find_cross_prodi_distractor,
)


def test_compute_cosine_similarity():
    v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v3 = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    assert pytest.approx(compute_cosine_similarity(v1, v2), 0.001) == 1.0
    assert pytest.approx(compute_cosine_similarity(v1, v3), 0.001) == 0.0
    assert compute_cosine_similarity(np.zeros(3), v1) == 0.0


def test_compute_ndcg():
    # Target: A dan B
    targets = {"MAS120", "MAS104"}
    
    # Perfect retrieval: A dan B di rank 1 dan 2
    perfect_retrieved = ["MAS120", "MAS104", "SIC101", "SIC102", "SIC103"]
    ndcg_perfect = compute_ndcg(perfect_retrieved, targets, k=5)
    assert pytest.approx(ndcg_perfect, 0.001) == 1.0

    # Partial retrieval: hanya A di rank 1
    partial_retrieved = ["MAS120", "SIC101", "SIC102", "SIC103", "SIC104"]
    ndcg_partial = compute_ndcg(partial_retrieved, targets, k=5)
    assert 0.0 < ndcg_partial < 1.0

    # Zero hit retrieval
    zero_retrieved = ["SIC101", "SIC102", "SIC103", "SIC104", "SIC105"]
    ndcg_zero = compute_ndcg(zero_retrieved, targets, k=5)
    assert ndcg_zero == 0.0


def test_find_cross_prodi_distractor():
    mock_catalog = [
        {"kode_mk": "MAS120", "nama_mk": "Pengantar Statistika", "program_studi": "Teknologi Sains Data"},
        {"kode_mk": "SIC203", "nama_mk": "Pengantar Kecerdasan Buatan", "program_studi": "Teknologi Sains Data"},
        {"kode_mk": "RND101", "nama_mk": "Pengantar Nanoteknologi", "program_studi": "Rekayasa Nanoteknologi"},
        {"kode_mk": "TEP201", "nama_mk": "Statistika Teknik", "program_studi": "Teknik Elektro"},
    ]

    # Target prodi TSD, query tentang statistika -> Harus pilih TEP201 dari Teknik Elektro
    distractor = find_cross_prodi_distractor(
        target_prodi="Teknologi Sains Data",
        target_text="Bahas materi statistika inferensial MAS120",
        catalog=mock_catalog,
        exclude_codes={"MAS120"},
    )

    assert distractor["kode_mk"] != "MAS120"
    assert "Teknologi Sains Data" not in distractor["program_studi"]
    assert distractor["kode_mk"] == "TEP201"


def test_benchmark_datasets_integrity():
    comp_path = Path("data/ground_truth/retrieval_complexity_benchmark.json")
    contam_path = Path("data/ground_truth/rag_contamination_eval.json")

    assert comp_path.exists(), "retrieval_complexity_benchmark.json harus ada"
    assert contam_path.exists(), "rag_contamination_eval.json harus ada"

    with open(comp_path, "r", encoding="utf-8") as f:
        comp_data = json.load(f)
    assert len(comp_data) == 120, "Harus tepat 120 butir soal complexity"

    with open(contam_path, "r", encoding="utf-8") as f:
        contam_data = json.load(f)
    assert len(contam_data) == 60, "Harus tepat 60 butir skenario contamination"


def test_exp030_artifacts_exist():
    out_dir = Path("docs/exp-030_multihop_retrieval_and_context_contamination")
    assert out_dir.exists(), "Direktori exp-030 harus ada"

    assert (out_dir / "README.md").exists()
    assert (out_dir / "benchmark_summary.md").exists()
    assert (out_dir / "benchmark_metrics.csv").exists()
    assert (out_dir / "benchmark_results.json").exists()
    assert (out_dir / "benchmark_raw_traces.jsonl").exists()

    with open(out_dir / "benchmark_results.json", "r", encoding="utf-8") as f:
        res = json.load(f)
    assert res["experiment_id"] == "exp-030"
    assert "complexity_summary" in res
    assert "contamination_summary" in res
