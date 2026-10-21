import json
from pathlib import Path

import numpy as np
import pytest

from scripts.evaluate_slm_llm_instruct_spectrum import (
    calculate_pareto_analysis,
    compute_cosine_similarity,
    compute_fact_recall,
    save_reports,
)


def test_compute_cosine_similarity():
    v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v3 = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    assert pytest.approx(compute_cosine_similarity(v1, v2), 0.001) == 1.0
    assert pytest.approx(compute_cosine_similarity(v1, v3), 0.001) == 0.0


def test_compute_fact_recall():
    gt = "Mata kuliah Agama Islam 1 (AGI101) berbobot 2 SKS di prodi Rekayasa Nanoteknologi."
    good_gen = "Mata kuliah AGI101 berbobot 2 SKS membahas nilai-nilai ketuhanan di Rekayasa Nanoteknologi."
    bad_gen = "Saya tidak mengetahui informasi mengenai hal tersebut."

    score_good = compute_fact_recall(good_gen, gt, source_id="AGI101")
    score_bad = compute_fact_recall(bad_gen, gt, source_id="AGI101")

    assert score_good >= 0.70
    assert score_bad <= 0.20


def test_calculate_pareto_analysis():
    summaries = [
        {
            "model": "model_fast_accurate",
            "tier": "SLM (Compact)",
            "eval_count": 10,
            "tool_selection_accuracy_pct": 95.0,
            "fact_recall_pct": 90.0,
            "semantic_similarity": 0.85,
            "p50_ttft_ms": 150.0,
            "mean_ttft_ms": 160.0,
            "mean_tps": 120.0,
            "peak_vram_mib": 2500,
            "truncation_rate_pct": 0.0,
            "context_efficiency_pct": 100.0,
        },
        {
            "model": "model_slow_poor",
            "tier": "Compact LLM",
            "eval_count": 10,
            "tool_selection_accuracy_pct": 50.0,
            "fact_recall_pct": 40.0,
            "semantic_similarity": 0.50,
            "p50_ttft_ms": 900.0,
            "mean_ttft_ms": 950.0,
            "mean_tps": 40.0,
            "peak_vram_mib": 6500,
            "truncation_rate_pct": 10.0,
            "context_efficiency_pct": 100.0,
        },
    ]

    pareto_res = calculate_pareto_analysis(summaries)
    assert len(pareto_res) == 2

    fast_acc = next(s for s in pareto_res if s["model"] == "model_fast_accurate")
    slow_poor = next(s for s in pareto_res if s["model"] == "model_slow_poor")

    assert fast_acc["is_pareto_frontier"] is True
    assert slow_poor["is_pareto_frontier"] is False
    assert fast_acc["pareto_score"] > slow_poor["pareto_score"]


def test_save_reports(tmp_path: Path):
    summaries = [
        {
            "model": "qwen2.5:3b",
            "tier": "SLM (Compact)",
            "eval_count": 5,
            "tool_selection_accuracy_pct": 90.0,
            "fact_recall_pct": 85.0,
            "semantic_similarity": 0.80,
            "p50_ttft_ms": 200.0,
            "mean_ttft_ms": 220.0,
            "mean_tps": 105.0,
            "peak_vram_mib": 2800,
            "truncation_rate_pct": 0.0,
            "context_efficiency_pct": 100.0,
            "quality_composite": 85.0,
            "speed_composite": 75.0,
            "memory_composite": 60.0,
            "pareto_score": 75.0,
            "is_pareto_frontier": True,
        }
    ]
    subcats = {
        "Silabus Wajib": {
            "count": 5,
            "best_model": "qwen2.5:3b",
            "best_fact_recall": 85.0,
            "best_tool_acc": 90.0,
        }
    }

    save_reports(summaries, subcats, tmp_path, dataset_count=5)

    assert (tmp_path / "benchmark_metrics.csv").exists()
    assert (tmp_path / "benchmark_results.json").exists()
    assert (tmp_path / "benchmark_summary.md").exists()
    assert (tmp_path / "README.md").exists()

    with open(tmp_path / "benchmark_results.json", "r", encoding="utf-8") as f:
        res_json = json.load(f)
        assert res_json["metadata"]["experiment_id"] == "exp-028"
        assert len(res_json["models"]) == 1
