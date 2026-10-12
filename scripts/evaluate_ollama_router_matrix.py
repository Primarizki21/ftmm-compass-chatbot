import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import ollama
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.evaluate_decider import EVAL_DATASET
from src.engine.decider import DecisionCategory, PROTOTYPES
OUTPUT_DIR = Path("docs/exp-017_ollama_embedding_classifier_head")
MODELS = ["nomic-embed-text", "qwen3-embedding:0.6b", "bge-m3"]
HEADS = ["Centroid", "Logistic", "kNN_1"]
CATEGORIES = [
    DecisionCategory.CHAT_ONLY,
    DecisionCategory.PREREQUISITE,
    DecisionCategory.SKS_CALCULATION,
    DecisionCategory.DEGREE_PLANNING,
    DecisionCategory.COURSE_DISCOVERY,
]
CAT_VALUES = [c.value for c in CATEGORIES]
CAT2IDX = {c.value: i for i, c in enumerate(CATEGORIES)}


def get_gpu_vram_mib() -> float:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
            encoding="utf-8",
        )
        return float(out.strip().split("\n")[0])
    except Exception:
        return 0.0


def embed_batch_ollama(model_name: str, texts: list[str]) -> tuple[np.ndarray, list[float]]:
    vecs = []
    latencies = []
    for text in texts:
        t0 = time.perf_counter()
        resp = ollama.embed(model=model_name, input=text)
        dt = (time.perf_counter() - t0) * 1000.0
        vec = np.array(resp["embeddings"][0], dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        vecs.append(vec)
        latencies.append(dt)
    return np.array(vecs, dtype=np.float32), latencies


def prepare_exemplars(model_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, LogisticRegression]:
    proto_texts = []
    proto_labels = []
    for cat in CATEGORIES:
        for text in PROTOTYPES[cat]:
            proto_texts.append(text)
            proto_labels.append(CAT2IDX[cat.value])

    print(f"  [{model_name}] Encoding {len(proto_texts)} exemplar prototypes...")
    proto_vecs, _ = embed_batch_ollama(model_name, proto_texts)
    proto_labels_arr = np.array(proto_labels, dtype=np.int32)

    # 1. Centroids Head
    centroids = []
    for cat in CATEGORIES:
        idx = CAT2IDX[cat.value]
        mask = proto_labels_arr == idx
        cat_mean = np.mean(proto_vecs[mask], axis=0)
        c_norm = np.linalg.norm(cat_mean)
        if c_norm > 0:
            cat_mean = cat_mean / c_norm
        centroids.append(cat_mean)
    centroid_matrix = np.array(centroids, dtype=np.float32)

    # 2. Logistic Regression Head
    clf = LogisticRegression(C=1.0, max_iter=500, random_state=42)
    clf.fit(proto_vecs, proto_labels_arr)

    return proto_vecs, proto_labels_arr, centroid_matrix, clf


def run_classifier_head(
    head_type: str,
    q_vec: np.ndarray,
    centroid_matrix: np.ndarray,
    proto_vecs: np.ndarray,
    proto_labels: np.ndarray,
    clf: LogisticRegression,
) -> tuple[str, float, float, float]:
    t0 = time.perf_counter()
    if head_type == "Centroid":
        scores = np.dot(centroid_matrix, q_vec)
        sorted_indices = np.argsort(scores)[::-1]
        best_idx = sorted_indices[0]
        second_idx = sorted_indices[1]
        pred_cat = CAT_VALUES[best_idx]
        conf = float(scores[best_idx])
        margin = float(scores[best_idx] - scores[second_idx])
    elif head_type == "Logistic":
        probas = clf.predict_proba(q_vec.reshape(1, -1))[0]
        sorted_indices = np.argsort(probas)[::-1]
        best_idx = sorted_indices[0]
        second_idx = sorted_indices[1]
        pred_cat = CAT_VALUES[clf.classes_[best_idx]]
        conf = float(probas[best_idx])
        margin = float(probas[best_idx] - probas[second_idx])
    elif head_type == "kNN_1":
        sims = np.dot(proto_vecs, q_vec)
        best_proto_idx = np.argmax(sims)
        best_cat_idx = proto_labels[best_proto_idx]
        pred_cat = CAT_VALUES[best_cat_idx]
        conf = float(sims[best_proto_idx])
        
        # Max sim in best class vs max sim in any other class
        mask_same = proto_labels == best_cat_idx
        mask_other = proto_labels != best_cat_idx
        max_same = np.max(sims[mask_same])
        max_other = np.max(sims[mask_other]) if np.any(mask_other) else 0.0
        margin = float(max_same - max_other)
    else:
        raise ValueError(f"Unknown head type: {head_type}")
    dt = (time.perf_counter() - t0) * 1000.0
    return pred_cat, conf, margin, dt


def evaluate_variant(
    model_name: str,
    head_type: str,
    query_vecs: np.ndarray,
    embed_latencies: list[float],
    centroid_matrix: np.ndarray,
    proto_vecs: np.ndarray,
    proto_labels: np.ndarray,
    clf: LogisticRegression,
    testbed: list[dict[str, Any]],
) -> dict[str, Any]:
    traces = []
    correct_count = 0
    correct_by_style = {"Formal": 0, "Slang": 0, "Implicit": 0}
    total_by_style = {"Formal": 0, "Slang": 0, "Implicit": 0}
    correct_by_cat = {c.value: 0 for c in CATEGORIES}
    total_by_cat = {c.value: 0 for c in CATEGORIES}
    margins = []
    head_latencies = []
    total_latencies = []

    for i, item in enumerate(testbed):
        q_vec = query_vecs[i]
        emb_lat = embed_latencies[i]
        gt_cat = item["category"]
        style = item["style"]

        pred_cat, conf, margin, head_lat = run_classifier_head(
            head_type, q_vec, centroid_matrix, proto_vecs, proto_labels, clf
        )

        is_correct = pred_cat == gt_cat
        if is_correct:
            correct_count += 1
            correct_by_style[style] += 1
            correct_by_cat[gt_cat] += 1

        total_by_style[style] += 1
        total_by_cat[gt_cat] += 1
        margins.append(margin)
        head_latencies.append(head_lat)
        total_latencies.append(emb_lat + head_lat)

        traces.append({
            "variant": f"{model_name}_{head_type}",
            "model": model_name,
            "head": head_type,
            "query_id": item["id"],
            "query": item["query"],
            "ground_truth": gt_cat,
            "prediction": pred_cat,
            "is_correct": is_correct,
            "style": style,
            "confidence": round(conf, 4),
            "margin": round(margin, 4),
            "embed_latency_ms": round(emb_lat, 2),
            "head_latency_ms": round(head_lat, 4),
            "total_latency_ms": round(emb_lat + head_lat, 2),
        })

    total = len(testbed)
    accuracy = correct_count / total * 100.0
    style_acc = {k: correct_by_style[k] / total_by_style[k] * 100.0 for k in correct_by_style}
    cat_acc = {k: correct_by_cat[k] / total_by_cat[k] * 100.0 for k in correct_by_cat}

    return {
        "variant": f"{model_name} + {head_type}",
        "model": model_name,
        "head": head_type,
        "total_queries": total,
        "correct": correct_count,
        "accuracy": round(accuracy, 2),
        "style_acc": {k: round(v, 2) for k, v in style_acc.items()},
        "cat_acc": {k: round(v, 2) for k, v in cat_acc.items()},
        "avg_margin": round(float(np.mean(margins)), 4),
        "avg_embed_latency_ms": round(float(np.mean(embed_latencies)), 2),
        "p50_embed_latency_ms": round(float(np.median(embed_latencies)), 2),
        "avg_head_latency_ms": round(float(np.mean(head_latencies)), 4),
        "avg_total_latency_ms": round(float(np.mean(total_latencies)), 2),
        "p50_total_latency_ms": round(float(np.median(total_latencies)), 2),
        "traces": traces,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("  Exp-016: Ollama Native Embedding & Classifier Head Matrix Benchmark")
    print(f"  Testbed: {len(EVAL_DATASET)} queries (20 Formal, 20 Slang, 20 Implicit)")
    print(f"  Models: {MODELS}")
    print(f"  Heads:  {HEADS}")
    print("=" * 70)

    vram_start = get_gpu_vram_mib()
    all_results = []
    all_traces = []

    for model_name in MODELS:
        print(f"\n[MODEL] Processing {model_name}...")
        proto_vecs, proto_labels, centroid_matrix, clf = prepare_exemplars(model_name)
        
        print(f"  Embedding {len(EVAL_DATASET)} test queries with {model_name}...")
        query_vecs, embed_latencies = embed_batch_ollama(
            model_name, [item["query"] for item in EVAL_DATASET]
        )

        for head_type in HEADS:
            res = evaluate_variant(
                model_name,
                head_type,
                query_vecs,
                embed_latencies,
                centroid_matrix,
                proto_vecs,
                proto_labels,
                clf,
                EVAL_DATASET,
            )
            all_results.append(res)
            all_traces.extend(res["traces"])
            print(
                f"  -> {head_type:<10} | Acc: {res['accuracy']:5.1f}% "
                f"(Formal: {res['style_acc']['Formal']:4.1f}%, Slang: {res['style_acc']['Slang']:4.1f}%, Imp: {res['style_acc']['Implicit']:4.1f}%) "
                f"| Margin: {res['avg_margin']:.4f} | P50: {res['p50_total_latency_ms']:5.1f} ms"
            )

    vram_end = get_gpu_vram_mib()

    # 1. Write benchmark_metrics.csv
    csv_path = OUTPUT_DIR / "benchmark_metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Variant",
            "Model",
            "Classifier_Head",
            "Overall_Acc",
            "Formal_Acc",
            "Slang_Acc",
            "Implicit_Acc",
            "Avg_Margin",
            "P50_Latency_ms",
            "Avg_Latency_ms",
            "Avg_Embed_Latency_ms",
            "Avg_Head_Latency_ms",
        ])
        for r in all_results:
            writer.writerow([
                r["variant"],
                r["model"],
                r["head"],
                r["accuracy"],
                r["style_acc"]["Formal"],
                r["style_acc"]["Slang"],
                r["style_acc"]["Implicit"],
                r["avg_margin"],
                r["p50_total_latency_ms"],
                r["avg_total_latency_ms"],
                r["avg_embed_latency_ms"],
                r["avg_head_latency_ms"],
            ])

    # 2. Write benchmark_results.json
    json_path = OUTPUT_DIR / "benchmark_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    # 3. Write benchmark_raw_traces.jsonl
    jsonl_path = OUTPUT_DIR / "benchmark_raw_traces.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for trace in all_traces:
            f.write(json.dumps(trace, ensure_ascii=False) + "\n")

    # 4. Generate benchmark_summary.md
    summary_path = OUTPUT_DIR / "benchmark_summary.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# Benchmark Summary — Exp-016: Ollama Native Embedding & Classifier Head Matrix\n\n")
        f.write("- **Tanggal Pengujian:** 2026-10-12\n")
        f.write("- **Hardware Platform:** NVIDIA GeForce RTX 5050 Laptop GPU (8GB VRAM), WSL2 Linux\n")
        f.write(f"- **VRAM Footprint:** Start {vram_start:.1f} MiB / Peak {vram_end:.1f} MiB (Zero Python RAM / Zero Swap)\n")
        f.write(f"- **Testbed:** {len(EVAL_DATASET)} Kueri Terstratifikasi (20 Formal, 20 Slang, 20 Implisit)\n\n")
        f.write("## 1. Tabel Komparasi Utama (9 Varian: 3 Model × 3 Head)\n\n")
        f.write("| No | Embedding Model | Classifier Head | Overall Acc (%) | Formal Acc (%) | Slang Acc (%) | Implicit Acc (%) | Avg Margin | P50 Latensi (ms) |\n")
        f.write("|:--:|---|---|:---:|:---:|:---:|:---:|:---:|:---:|\n")
        for i, r in enumerate(all_results, start=1):
            f.write(
                f"| {i} | `{r['model']}` | **{r['head']}** | **{r['accuracy']}%** | "
                f"{r['style_acc']['Formal']}% | {r['style_acc']['Slang']}% | {r['style_acc']['Implicit']}% | "
                f"{r['avg_margin']:.4f} | {r['p50_total_latency_ms']:.1f} ms |\n"
            )
        f.write("\n## 2. Analisis Performa Berdasarkan Kategori Akademik FTMM\n\n")
        f.write("| Varian | CHAT_ONLY | PREREQUISITE | SKS_CALCULATION | DEGREE_PLANNING | COURSE_DISCOVERY |\n")
        f.write("|---|:---:|:---:|:---:|:---:|:---:|\n")
        for r in all_results:
            f.write(
                f"| `{r['variant']}` | {r['cat_acc']['CHAT_ONLY']}% | {r['cat_acc']['PREREQUISITE']}% | "
                f"{r['cat_acc']['SKS_CALCULATION']}% | {r['cat_acc']['DEGREE_PLANNING']}% | {r['cat_acc']['COURSE_DISCOVERY']}% |\n"
            )

    # 5. Generate README.md
    readme_path = OUTPUT_DIR / "README.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("# Exp-016: Ollama Native Embedding & Classifier Head Matrix\n\n")
        f.write("Eksperimen komparatif evaluasi 3 model embedding Ollama native (`nomic-embed-text`, `qwen3-embedding:0.6b`, `bge-m3`) dipadukan dengan 3 variasi arsitektur classifier head (`Centroid`, `Logistic Regression`, `k-NN`) untuk System-1 Decision Classifier FTMM COMPASS.\n")

    print(f"\n[SUCCESS] Benchmark Exp-016 selesai!")
    print(f"  Artifacts saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
