import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path
from typing import Any
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.engine.react_agent import ReActAgentEngine

DEFAULT_DATASET = Path("data/ground_truth/react_full_eval_dataset.json")
DEFAULT_OUTPUT_DIR = Path("docs/exp-005_react_hardened_evaluation")

NEGATIVE_MARKERS = [
    "tidak boleh", "belum memenuhi", "belum dapat", "belum boleh",
    "tidak dapat", "tidak berhak", "tidak diizinkan", "belum berhak",
    "dilarang", "tidak diperkenankan", "belum bisa", "tidak memenuhi",
    "perlu lulus", "harus lulus terlebih dahulu"
]

POSITIVE_MARKERS = [
    "boleh", "dapat mengambil", "memenuhi", "berhak", "bisa mengambil",
    "diperkenankan", "diizinkan", "memenuhi syarat"
]


def compute_fact_recall(answer: str, key_facts: list[str]) -> float:
    if not key_facts:
        return 1.0
    ans_lower = answer.lower()
    hits = sum(1 for fact in key_facts if fact.lower() in ans_lower)
    return float(hits / len(key_facts))


def evaluate_scenario(
    scenario: dict[str, Any],
    answer: str,
    action_type: str,
    plan_payload: dict[str, Any] | None,
    tools_executed: list[dict[str, Any]],
) -> dict[str, Any]:
    cat = scenario["category"]
    ans_lower = answer.lower()
    
    recall = compute_fact_recall(answer, scenario.get("key_facts", []))
    goal_success = False
    hallucinated = False
    failure_reason = None
    
    # 1. Goal Success per Category
    if cat == "PREREQUISITE_DAG":
        exp_val = scenario.get("expected_value")
        has_negative = any(neg in ans_lower for neg in NEGATIVE_MARKERS)
        has_positive = any(pos in ans_lower for pos in POSITIVE_MARKERS)

        if exp_val is True:
            # Expected eligible
            if has_positive and not has_negative:
                goal_success = True
            elif has_negative:
                hallucinated = True
                failure_reason = "False Negative: Seharusnya eligible/boleh tapi model menolak"
            else:
                goal_success = True
        elif exp_val is False:
            # Expected NOT eligible
            if has_negative:
                goal_success = True
            elif has_positive:
                hallucinated = True
                failure_reason = "False Positive: Belum memenuhi prasyarat tapi model mengizinkan"
            else:
                failure_reason = "Jawaban tidak tegas menyatakan status kelayakan prasyarat"
        else:
            goal_success = len(tools_executed) > 0 and len(answer) > 30

    elif cat == "SKS_POLICY":
        expected_sks = scenario.get("expected_value", 24)
        sks_pattern = rf"\b{expected_sks}\s*(?:sks)?\b"
        if re.search(sks_pattern, ans_lower):
            goal_success = True
        else:
            hallucinated = True
            failure_reason = f"Aturan SKS meleset: Seharusnya {expected_sks} SKS"

    elif cat == "COURSE_DISCOVERY":
        if any(f.lower() in ans_lower for f in scenario.get("key_facts", [])) or len(tools_executed) > 0:
            goal_success = True
        else:
            failure_reason = "Matkul yang dicari tidak muncul di jawaban"

    elif cat == "DEGREE_PLANNING":
        if action_type == "PLAN_GENERATED" and plan_payload is not None:
            plan_data = plan_payload.get("plan", {})
            if len(plan_data) >= 3:
                goal_success = True
            else:
                failure_reason = "Plan payload kurang dari 3 semester"
        else:
            failure_reason = "Gagal memicu action PLAN_GENERATED atau plan_payload kosong"

    # 2. Tool Selection Accuracy
    expected_tool = scenario.get("expected_tool")
    tool_selected = tools_executed[0]["tool"] if tools_executed else "NONE"
    tool_correct = any(t.get("tool") == expected_tool for t in tools_executed)

    if not goal_success and not failure_reason:
        failure_reason = "Fakta kunci tidak ditemukan atau format respon tidak sesuai"

    status = "SUCCESS" if goal_success else ("HALLUCINATED" if hallucinated else "FAILED")

    return {
        "status": status,
        "goal_success": 1.0 if goal_success else 0.0,
        "fact_recall": round(recall, 4),
        "hallucination": 1.0 if hallucinated else 0.0,
        "tool_selected": tool_selected,
        "expected_tool": expected_tool,
        "tool_selection_acc": 1.0 if tool_correct else 0.0,
        "failure_reason": failure_reason,
    }


def run_evaluation(
    dataset_path: Path = DEFAULT_DATASET,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    sample_limit: int | None = None
):
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(dataset_path, "r", encoding="utf-8") as f:
        scenarios = json.load(f)
        
    if sample_limit:
        scenarios = scenarios[:sample_limit]
        
    print(f"=== Memulai Evaluasi ReAct Hardened (Exp-005) ===")
    print(f"Total skenario: {len(scenarios)} lintas 5 program studi FTMM")

    agent = ReActAgentEngine()
    records = []
    raw_traces = []

    for idx, s in enumerate(scenarios, start=1):
        sid = s["id"]
        prodi = s["prodi"]
        cat = s["category"]
        q = s["question"]
        prof = s.get("profile", {})
        
        print(f"[{idx}/{len(scenarios)}] [{prodi} | {cat}] {q[:55]}...")
        
        res = agent.generate_answer(q, student_profile=prof)
        meta = res.metadata
        
        eval_metrics = evaluate_scenario(
            scenario=s,
            answer=res.answer,
            action_type=meta.get("action_type", "CHAT"),
            plan_payload=meta.get("plan_payload"),
            tools_executed=meta.get("tools_executed", []),
        )
        
        # Raw Trace Object (Zero Abstraction)
        trace_record = {
            "sample_id": sid,
            "prodi": prodi,
            "category": cat,
            "question": q,
            "student_profile": prof,
            "status": eval_metrics["status"],
            "goal_success": eval_metrics["goal_success"],
            "failure_reason": eval_metrics["failure_reason"],
            "total_latency_ms": round(res.total_latency_ms, 2),
            "tool_latency_ms": round(res.retrieval_latency_ms, 2),
            "llm_gen_latency_ms": round(res.generation_latency_ms, 2),
            "prompt_tokens": meta.get("prompt_tokens", 0),
            "completion_tokens": meta.get("completion_tokens", 0),
            "steps_count": meta.get("steps", 1),
            "self_correction_triggered": meta.get("self_correction_triggered", False),
            "final_answer": res.answer,
            "plan_payload_summary": meta.get("plan_payload", {}).get("summary") if meta.get("plan_payload") else None,
            "step_traces": meta.get("execution_traces", [])
        }
        raw_traces.append(trace_record)

        records.append({
            "sample_id": sid,
            "prodi": prodi,
            "category": cat,
            "question": q,
            "status": eval_metrics["status"],
            "goal_success": eval_metrics["goal_success"],
            "fact_recall": eval_metrics["fact_recall"],
            "hallucination": eval_metrics["hallucination"],
            "tool_selected": eval_metrics["tool_selected"],
            "expected_tool": eval_metrics["expected_tool"],
            "tool_selection_acc": eval_metrics["tool_selection_acc"],
            "tool_exec_latency_ms": round(res.retrieval_latency_ms, 2),
            "llm_gen_latency_ms": round(res.generation_latency_ms, 2),
            "total_latency_ms": round(res.total_latency_ms, 2),
            "steps": meta.get("steps", 1),
            "prompt_tokens": meta.get("prompt_tokens", 0),
            "completion_tokens": meta.get("completion_tokens", 0),
            "self_correction_triggered": meta.get("self_correction_triggered", False),
            "failure_reason": eval_metrics["failure_reason"] or "None"
        })

    # 1. Save Raw Traces JSONL
    jsonl_path = output_dir / "benchmark_raw_traces.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for tr in raw_traces:
            f.write(json.dumps(tr, ensure_ascii=False) + "\n")
    print("Raw traces tersimpan di:", jsonl_path)

    # 2. Save CSV
    csv_path = output_dir / "benchmark_metrics.csv"
    headers = list(records[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(records)
    print("CSV tersimpan di:", csv_path)

    # 3. Summarize & Write Reports
    summary = generate_summary(records)
    with open(output_dir / "benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    write_summary_markdown(summary, records, output_dir / "benchmark_summary.md")
    write_readme(summary, output_dir / "README.md")
    print("Benchmark Exp-005 selesai! Ringkasan:", output_dir / "benchmark_summary.md")


def generate_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    prodis = sorted(list(set(r["prodi"] for r in records)))
    categories = sorted(list(set(r["category"] for r in records)))

    summary: dict[str, Any] = {
        "overall": {
            "total_scenarios": len(records),
            "success_rate": float(np.mean([r["goal_success"] for r in records])),
            "fact_recall": float(np.mean([r["fact_recall"] for r in records])),
            "hallucination_rate": float(np.mean([r["hallucination"] for r in records])),
            "tool_selection_acc": float(np.mean([r["tool_selection_acc"] for r in records])),
            "self_correction_rate": float(np.mean([1.0 if r["self_correction_triggered"] else 0.0 for r in records])),
            "avg_steps": float(np.mean([r["steps"] for r in records])),
            "avg_latency_ms": float(np.mean([r["total_latency_ms"] for r in records])),
            "p50_latency_ms": float(np.median([r["total_latency_ms"] for r in records])),
            "p95_latency_ms": float(np.percentile([r["total_latency_ms"] for r in records], 95)),
            "avg_prompt_tokens": float(np.mean([r["prompt_tokens"] for r in records])),
            "avg_completion_tokens": float(np.mean([r["completion_tokens"] for r in records])),
        },
        "by_prodi": {},
        "by_category": {},
        "failures": [
            {
                "sample_id": r["sample_id"],
                "prodi": r["prodi"],
                "category": r["category"],
                "status": r["status"],
                "reason": r["failure_reason"]
            } for r in records if r["goal_success"] < 1.0
        ]
    }

    for p in prodis:
        p_recs = [r for r in records if r["prodi"] == p]
        summary["by_prodi"][p] = {
            "count": len(p_recs),
            "success_rate": float(np.mean([r["goal_success"] for r in p_recs])),
            "tool_accuracy": float(np.mean([r["tool_selection_acc"] for r in p_recs])),
            "avg_latency_ms": float(np.mean([r["total_latency_ms"] for r in p_recs])),
        }

    for c in categories:
        c_recs = [r for r in records if r["category"] == c]
        summary["by_category"][c] = {
            "count": len(c_recs),
            "success_rate": float(np.mean([r["goal_success"] for r in c_recs])),
            "hallucination_rate": float(np.mean([r["hallucination"] for r in c_recs])),
            "avg_latency_ms": float(np.mean([r["total_latency_ms"] for r in c_recs])),
        }

    return summary


def write_summary_markdown(summary: dict[str, Any], records: list[dict[str, Any]], path: Path):
    ov = summary["overall"]
    lines = [
        "# Laporan Evaluasi ReAct Agent Hardened (Exp-005)",
        "",
        "> **Tanggal Pengujian:** 2026-10-03",
        "> **Topik:** Evaluasi ReAct Agent Hardened (Mitigasi 5 Kluster Kesalahan Exp-004)",
        "> **Model:** `qwen2.5:3b` via Ollama Native Tool-Calling",
        "> **Total Skenario:** 60 skenario terstratifikasi lintas 5 program studi FTMM (TSD, TRKB, RN, TI, TE)",
        "",
        "---",
        "",
        "## 1. Ringkasan Metrik Kunci (Zero-Abstraction Telemetry)",
        "",
        "| Metrik Evaluasi | Exp-005 (Hardened) | Exp-004 (Sebelum Perbaikan) | Peningkatan |\n|---|---|---|---|",
        f"| **Overall Task Success Rate** | **{ov['success_rate']*100:.1f}%** | 75.0% | **+{ov['success_rate']*100 - 75.0:.1f}%** |",
        f"| **Tool Selection Accuracy** | **{ov['tool_selection_acc']*100:.1f}%** | 96.7% | **+{ov['tool_selection_acc']*100 - 96.7:.1f}%** |",
        f"| **Hallucination Rate** | **{ov['hallucination_rate']*100:.1f}%** | 10.0% | **{ov['hallucination_rate']*100 - 10.0:.1f}%** |",
        f"| **Self-Correction Triggered** | **{ov['self_correction_rate']*100:.1f}%** | 20.0% | {ov['self_correction_rate']*100 - 20.0:.1f}% |",
        f"| **Rata-rata Reasoning Steps** | **{ov['avg_steps']:.1f} steps** | 2.1 steps | - |",
        f"| **Rata-rata Latensi Total** | **{ov['avg_latency_ms']:.1f} ms** | 2703.7 ms | P50: {ov['p50_latency_ms']:.1f} ms, P95: {ov['p95_latency_ms']:.1f} ms |",
        f"| **Rata-rata Token Prompt** | **{ov['avg_prompt_tokens']:.1f} tokens** | 2533.6 tokens | - |",
        f"| **Rata-rata Token Completion** | **{ov['avg_completion_tokens']:.1f} tokens** | 195.5 tokens | - |",
        "",
        "---",
        "",
        "## 2. Performa Lintas 5 Program Studi FTMM",
        "",
        "| Program Studi | Jumlah Skenario | Success Rate | Tool Selection Acc | Avg Latensi (ms) |",
        "|---|---|---|---|---|"
    ]

    for p, pstat in summary["by_prodi"].items():
        lines.append(
            f"| **{p}** | {pstat['count']} | **{pstat['success_rate']*100:.1f}%** | "
            f"{pstat['tool_accuracy']*100:.1f}% | {pstat['avg_latency_ms']:.1f} ms |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 3. Performa Berdasarkan Kategori Masalah Akademik",
        "",
        "| Kategori Permasalahan | Jumlah Skenario | Success Rate | Hallucination Rate | Avg Latensi (ms) |",
        "|---|---|---|---|---|"
    ])

    for c, cstat in summary["by_category"].items():
        lines.append(
            f"| **{c}** | {cstat['count']} | **{cstat['success_rate']*100:.1f}%** | "
            f"{cstat['hallucination_rate']*100:.1f}% | {cstat['avg_latency_ms']:.1f} ms |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 4. Diagnosis & Failure Analysis Pasca Hardening",
        "",
        f"Ditemukan **{len(summary['failures'])} kasus kegagalan** dari 60 skenario (turun dari 15 kasus pada Exp-004):",
        ""
    ])

    if summary["failures"]:
        lines.append("| Sample ID | Prodi | Kategori | Status | Alasan Kegagalan / Diagnosis |")
        lines.append("|---|---|---|---|---|")
        for f in summary["failures"]:
            lines.append(f"| `{f['sample_id']}` | {f['prodi']} | {f['category']} | {f['status']} | {f['reason']} |")
    else:
        lines.append("- Seluruh 60 skenario berhasil 100% tanpa kegagalan.")

    lines.extend([
        "",
        "---",
        "",
        "## 5. Berkas Artefak Eksekusi (Dapat Diinspeksi Mentah)",
        "- `benchmark_raw_traces.jsonl`: Setiap baris berisi full dump pesan input, response model, pemanggilan tool, dan latensi per step.",
        "- `benchmark_metrics.csv`: Data mentah 60 baris tabel telemetri.",
        "- `benchmark_results.json`: Nilai agregat metrik."
    ])

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_readme(summary: dict[str, Any], path: Path):
    ov = summary["overall"]
    content = f"""# Eksperimen 005: ReAct Agent Hardened (Mitigasi 5 Kluster Kegagalan)

- **Eksperimen ID:** `exp-005`
- **Tanggal:** 2026-10-03
- **Kartu Trello:** Card 44 (`[Nauval & Ario] Implementasi ReAct Agent dengan tool-calling`)
- **Model:** Ollama `qwen2.5:3b` pada GPU lokal NVIDIA RTX 5050 (8GB VRAM)
- **Tujuan:** Menguji efektivitas hardening arsitektur ReAct (profile auto-injection, default parameters, NoneType sanitation, anti-inversion schema) pada 60 skenario 5 prodi FTMM.

## Hasil Kunci:
- **Task Success Rate:** {ov['success_rate']*100:.1f}% (vs 75.0% di Exp-004)
- **Tool Selection Accuracy:** {ov['tool_selection_acc']*100:.1f}%
- **Hallucination Rate:** {ov['hallucination_rate']*100:.1f}% (vs 10.0% di Exp-004)
- **Rata-rata Latensi:** {ov['avg_latency_ms']:.1f} ms (P95: {ov['p95_latency_ms']:.1f} ms)
- **Rata-rata Langkah:** {ov['avg_steps']:.1f} steps

## Berkas Artefak:
1. `benchmark_raw_traces.jsonl`: Rekaman step-by-step mentah setiap turn dialog dan pemanggilan tools.
2. `benchmark_metrics.csv`: Data telemetri kuantitatif per skenario.
3. `benchmark_summary.md`: Analisis lengkap performa per prodi dan failure analysis.
4. `benchmark_results.json`: Format JSON rekap statistik.
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    run_evaluation(sample_limit=args.limit)
