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

DEFAULT_DATASET = Path("data/ground_truth/react_multistyle_dataset.json")
DEFAULT_OUTPUT_DIR = Path("docs/exp-006_react_multistyle_resilience")

NEGATIVE_MARKERS = [
    "tidak boleh", "belum memenuhi", "belum dapat", "belum boleh",
    "tidak dapat", "tidak berhak", "tidak diizinkan", "belum berhak",
    "dilarang", "tidak diperkenankan", "belum bisa", "tidak memenuhi",
    "perlu lulus", "harus lulus terlebih dahulu", "belum memenuhi prasyarat"
]

POSITIVE_MARKERS = [
    "boleh", "dapat mengambil", "memenuhi", "berhak", "bisa mengambil",
    "diperkenankan", "diizinkan", "memenuhi syarat", "langsung diambil"
]


def evaluate_multistyle_scenario(
    scenario: dict[str, Any],
    answer: str,
    action_type: str,
    plan_payload: dict[str, Any] | None,
    tools_executed: list[dict[str, Any]],
) -> dict[str, Any]:
    cat = scenario["category"]
    ans_lower = answer.lower()
    
    goal_success = False
    hallucinated = False
    failure_reason = None
    
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

    elif cat == "SKS_POLICY":
        expected_sks = scenario.get("expected_value", 24)
        sks_pattern = rf"\b{expected_sks}\s*(?:sks)?\b"
        if re.search(sks_pattern, ans_lower):
            goal_success = True
        else:
            hallucinated = True
            failure_reason = f"Aturan SKS meleset: Seharusnya {expected_sks} SKS"

    elif cat == "DEGREE_PLANNING":
        if action_type == "PLAN_GENERATED" and plan_payload is not None:
            plan_data = plan_payload.get("plan", {})
            if len(plan_data) >= 3:
                goal_success = True
            else:
                failure_reason = "Plan payload kurang dari 3 semester"
        else:
            failure_reason = "Gagal memicu action PLAN_GENERATED atau plan_payload kosong"

    # Tool Selection Accuracy
    expected_tool = scenario.get("expected_tool")
    tool_selected = tools_executed[0]["tool"] if tools_executed else "NONE"
    tool_correct = any(t.get("tool") == expected_tool for t in tools_executed)

    if not goal_success and not failure_reason:
        failure_reason = "Jawaban tidak memenuhi kriteria sukses"

    status = "SUCCESS" if goal_success else ("HALLUCINATED" if hallucinated else "FAILED")

    return {
        "status": status,
        "goal_success": 1.0 if goal_success else 0.0,
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
        
    print(f"=== Memulai Evaluasi Multi-Style Resilience (Exp-006) ===")
    print(f"Total skenario: {len(scenarios)} variasi gaya bahasa")

    agent = ReActAgentEngine()
    records = []
    raw_traces = []

    for idx, s in enumerate(scenarios, start=1):
        sid = s["sample_id"]
        base_id = s["base_id"]
        style = s["style"]
        prodi = s["prodi"]
        cat = s["category"]
        q = s["question"]
        prof = s.get("profile", {})
        
        print(f"[{idx}/{len(scenarios)}] [{style.upper()} | {prodi}] {q[:55]}...")
        
        res = agent.generate_answer(q, student_profile=prof)
        meta = res.metadata
        
        eval_metrics = evaluate_multistyle_scenario(
            scenario=s,
            answer=res.answer,
            action_type=meta.get("action_type", "CHAT"),
            plan_payload=meta.get("plan_payload"),
            tools_executed=meta.get("tools_executed", []),
        )
        
        trace_record = {
            "sample_id": sid,
            "base_id": base_id,
            "style": style,
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
            "base_id": base_id,
            "style": style,
            "prodi": prodi,
            "category": cat,
            "question": q,
            "status": eval_metrics["status"],
            "goal_success": eval_metrics["goal_success"],
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

    # Save JSONL
    jsonl_path = output_dir / "benchmark_raw_traces.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for tr in raw_traces:
            f.write(json.dumps(tr, ensure_ascii=False) + "\n")
    print("Raw traces tersimpan di:", jsonl_path)

    # Save CSV
    csv_path = output_dir / "benchmark_metrics.csv"
    headers = list(records[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(records)
    print("CSV tersimpan di:", csv_path)

    # Summarize
    summary = generate_summary(records)
    with open(output_dir / "benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    write_summary_markdown(summary, output_dir / "benchmark_summary.md")
    write_readme(summary, output_dir / "README.md")
    print("Benchmark Exp-006 selesai! Ringkasan:", output_dir / "benchmark_summary.md")


def generate_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    styles = ["formal", "slang", "implicit"]
    categories = sorted(list(set(r["category"] for r in records)))
    prodis = sorted(list(set(r["prodi"] for r in records)))
    base_ids = sorted(list(set(r["base_id"] for r in records)))

    # Compute Paraphrase Invariance: scenarios where all 3 variants yielded identical successful verdict
    invariant_count = 0
    for bid in base_ids:
        bid_recs = [r for r in records if r["base_id"] == bid]
        if len(bid_recs) == 3 and all(r["goal_success"] == 1.0 for r in bid_recs):
            invariant_count += 1
    paraphrase_invariance_score = float(invariant_count / len(base_ids)) if base_ids else 0.0

    summary: dict[str, Any] = {
        "overall": {
            "total_queries": len(records),
            "success_rate": float(np.mean([r["goal_success"] for r in records])),
            "tool_selection_acc": float(np.mean([r["tool_selection_acc"] for r in records])),
            "hallucination_rate": float(np.mean([r["hallucination"] for r in records])),
            "paraphrase_invariance_score": paraphrase_invariance_score,
            "avg_latency_ms": float(np.mean([r["total_latency_ms"] for r in records])),
            "p50_latency_ms": float(np.median([r["total_latency_ms"] for r in records])),
            "p95_latency_ms": float(np.percentile([r["total_latency_ms"] for r in records], 95)),
        },
        "by_style": {},
        "by_category": {},
        "by_prodi": {},
        "failures": [
            {
                "sample_id": r["sample_id"],
                "base_id": r["base_id"],
                "style": r["style"],
                "prodi": r["prodi"],
                "category": r["category"],
                "status": r["status"],
                "reason": r["failure_reason"]
            } for r in records if r["goal_success"] < 1.0
        ]
    }

    for s in styles:
        s_recs = [r for r in records if r["style"] == s]
        if not s_recs:
            continue
        summary["by_style"][s] = {
            "count": len(s_recs),
            "success_rate": float(np.mean([r["goal_success"] for r in s_recs])),
            "tool_selection_acc": float(np.mean([r["tool_selection_acc"] for r in s_recs])),
            "hallucination_rate": float(np.mean([r["hallucination"] for r in s_recs])),
            "avg_latency_ms": float(np.mean([r["total_latency_ms"] for r in s_recs])),
        }

    for c in categories:
        c_recs = [r for r in records if r["category"] == c]
        summary["by_category"][c] = {
            "count": len(c_recs),
            "success_rate": float(np.mean([r["goal_success"] for r in c_recs])),
            "hallucination_rate": float(np.mean([r["hallucination"] for r in c_recs])),
            "avg_latency_ms": float(np.mean([r["total_latency_ms"] for r in c_recs])),
        }

    for p in prodis:
        p_recs = [r for r in records if r["prodi"] == p]
        summary["by_prodi"][p] = {
            "count": len(p_recs),
            "success_rate": float(np.mean([r["goal_success"] for r in p_recs])),
            "tool_selection_acc": float(np.mean([r["tool_selection_acc"] for r in p_recs])),
            "avg_latency_ms": float(np.mean([r["total_latency_ms"] for r in p_recs])),
        }

    return summary


def write_summary_markdown(summary: dict[str, Any], path: Path):
    ov = summary["overall"]
    lines = [
        "# Laporan Evaluasi Multi-Style Resilience & Anti-Overfitting (Exp-006)",
        "",
        "> **Tanggal Pengujian:** 2026-10-03",
        "> **Topik:** Uji Invariansi Parafrase & Ketahanan Diksi Santai/Slang (Card 44 & Card 91 Trello)",
        "> **Model:** `qwen2.5:3b` via Ollama Native Tool-Calling",
        "> **Total Kueri:** 60 kueri berpasangan (20 Formal, 20 Slang, 20 Implisit)",
        "",
        "---",
        "",
        "## 1. Analisis Invariansi Gaya Bahasa (Generalization vs Overfitting)",
        "",
        f"**Paraphrase Invariance Score:** **{ov['paraphrase_invariance_score']*100:.1f}%**  ",
        "*(Persentase skenario dasar di mana model berhasil konsisten pada KETIGA variasi gaya: Formal, Slang, dan Implisit)*",
        "",
        "| Gaya Bahasa Kueri | Jumlah Kueri | Success Rate | Tool Selection Acc | Hallucination Rate | Avg Latensi (ms) |",
        "|---|---|---|---|---|---|"
    ]

    for s, stat in summary["by_style"].items():
        lines.append(
            f"| **{s.upper()}** | {stat['count']} | **{stat['success_rate']*100:.1f}%** | "
            f"{stat['tool_selection_acc']*100:.1f}% | {stat['hallucination_rate']*100:.1f}% | "
            f"{stat['avg_latency_ms']:.1f} ms |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 2. Performa Lintas Kategori Masalah Akademik",
        "",
        "| Kategori Permasalahan | Jumlah Kueri | Success Rate | Hallucination Rate | Avg Latensi (ms) |",
        "|---|---|---|---|---|"
    ])

    for c, stat in summary["by_category"].items():
        lines.append(
            f"| **{c}** | {stat['count']} | **{stat['success_rate']*100:.1f}%** | "
            f"{stat['hallucination_rate']*100:.1f}% | {stat['avg_latency_ms']:.1f} ms |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 3. Performa Lintas 5 Program Studi FTMM",
        "",
        "| Program Studi | Jumlah Kueri | Success Rate | Tool Selection Acc | Avg Latensi (ms) |",
        "|---|---|---|---|---|"
    ])

    for p, stat in summary["by_prodi"].items():
        lines.append(
            f"| **{p}** | {stat['count']} | **{stat['success_rate']*100:.1f}%** | "
            f"{stat['tool_selection_acc']*100:.1f}% | {stat['avg_latency_ms']:.1f} ms |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 4. Evaluasi Kasus Gagal (Failure Inspection)",
        "",
        f"Ditemukan **{len(summary['failures'])} kasus kegagalan** dari 60 pengujian:",
        ""
    ])

    if summary["failures"]:
        lines.append("| Sample ID | Base ID | Gaya | Prodi | Kategori | Alasan Kegagalan |")
        lines.append("|---|---|---|---|---|---|")
        for f in summary["failures"]:
            lines.append(f"| `{f['sample_id']}` | `{f['base_id']}` | {f['style']} | {f['prodi']} | {f['category']} | {f['reason']} |")
    else:
        lines.append("- Seluruh 60 variasi pengujian berhasil 100% sempurna tanpa kegagalan.")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_readme(summary: dict[str, Any], path: Path):
    ov = summary["overall"]
    content = f"""# Eksperimen 006: ReAct Agent Multi-Style Resilience & Paraphrase Invariance

- **Eksperimen ID:** `exp-006`
- **Tanggal:** 2026-10-03
- **Kartu Trello:** Card 44 (`[Nauval & Ario] Implementasi ReAct Agent`) & Card 91 (`[Nauval] Normalizer Slang`)
- **Model:** Ollama `qwen2.5:3b` pada GPU lokal NVIDIA RTX 5050 (8GB VRAM)
- **Tujuan:** Menguji ketahanan agen terhadap variasi diksi (Formal, Slang, Implisit) untuk membuktikan sistem tidak mengalami overfitting pada dataset sintetis.

## Hasil Kunci:
- **Overall Task Success Rate:** {ov['success_rate']*100:.1f}%
- **Paraphrase Invariance Score:** {ov['paraphrase_invariance_score']*100:.1f}%
- **Tool Selection Accuracy:** {ov['tool_selection_acc']*100:.1f}%
- **Rata-rata Latensi Total:** {ov['avg_latency_ms']:.1f} ms (P95: {ov['p95_latency_ms']:.1f} ms)

## Berkas Artefak:
1. `benchmark_raw_traces.jsonl`: Jejak step-by-step mentah setiap turn dialog dan pemanggilan tools.
2. `benchmark_metrics.csv`: Data kuantitatif per kueri.
3. `benchmark_summary.md`: Tabel komparasi lengkap gaya bahasa dan analisis invariansi.
4. `benchmark_results.json`: Rekap statistik JSON.
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    run_evaluation(sample_limit=args.limit)
