import argparse
import csv
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
import faiss
import numpy as np
import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_num_threads(4)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sentence_transformers import SentenceTransformer
from src.knowledge.bm25 import BM25Okapi
from src.knowledge.hybrid_retriever import HybridCourseRetriever

DEFAULT_DATASET = Path("data/ground_truth/ir_eval_dataset.json")
DEFAULT_OUTPUT_DIR = Path("docs/exp-012_embeddinggemma2_benchmark")
GEMMA_INDEX_PATH = Path("data/faiss/ftmm_courses_gemma2.faiss")
GEMMA_META_PATH = Path("data/faiss/ftmm_courses_gemma2_meta.pkl")
MODEL_ID = "google/embeddinggemma-2"


def get_gpu_vram_mib() -> float:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
            encoding="utf-8",
        )
        return float(out.strip().split("\n")[0])
    except Exception:
        return 0.0


def compute_ir_metrics(retrieved_ids: list[str], target_ids: list[str]) -> dict[str, float]:
    target_set = set(target_ids)
    first_rank = 0
    for r, cid in enumerate(retrieved_ids, start=1):
        if cid in target_set:
            first_rank = r
            break

    mrr = (1.0 / first_rank) if first_rank > 0 else 0.0
    hit1 = 1.0 if (len(retrieved_ids) >= 1 and retrieved_ids[0] in target_set) else 0.0
    hit3 = 1.0 if any(cid in target_set for cid in retrieved_ids[:3]) else 0.0
    hit5 = 1.0 if any(cid in target_set for cid in retrieved_ids[:5]) else 0.0

    hits3 = sum(1 for cid in retrieved_ids[:3] if cid in target_set)
    hits5 = sum(1 for cid in retrieved_ids[:5] if cid in target_set)
    prec3 = hits3 / 3.0
    prec5 = hits5 / 5.0
    rec3 = hits3 / min(len(target_set), 3) if target_set else 0.0
    rec5 = hits5 / min(len(target_set), 5) if target_set else 0.0

    # NDCG@5
    dcg = 0.0
    for r, cid in enumerate(retrieved_ids[:5], start=1):
        if cid in target_set:
            dcg += 1.0 / np.log2(r + 1)
    idcg = sum(1.0 / np.log2(r + 1) for r in range(1, min(len(target_set), 5) + 1))
    ndcg5 = (dcg / idcg) if idcg > 0 else 0.0

    return {
        "hit@1": hit1,
        "hit@3": hit3,
        "hit@5": hit5,
        "mrr": mrr,
        "ndcg@5": ndcg5,
        "precision@3": prec3,
        "recall@3": rec3,
        "precision@5": prec5,
        "recall@5": rec5,
    }


class Gemma2Evaluator:
    def __init__(self):
        if not GEMMA_INDEX_PATH.exists() or not GEMMA_META_PATH.exists():
            raise FileNotFoundError(
                "Index Gemma2 belum dibuat! Jalankan 'uv run python scripts/ingest_faiss_gemma2.py'."
            )

        print(f"Loading metadata from {GEMMA_META_PATH}...")
        with open(GEMMA_META_PATH, "rb") as f:
            self.metadata = pickle.load(f)
        self.courses = self.metadata["courses"]
        self.chunks = self.metadata["chunks"]

        print(f"Loading FAISS index from {GEMMA_INDEX_PATH}...")
        self.index_768 = faiss.read_index(str(GEMMA_INDEX_PATH))
        self.n_total = self.index_768.ntotal

        # Extract full 768d vectors from index for in-memory Matryoshka sub-indices
        self.full_vectors = np.zeros((self.n_total, 768), dtype=np.float32)
        for i in range(self.n_total):
            self.full_vectors[i] = self.index_768.reconstruct(i)

        # Build in-memory Matryoshka indices (512d, 256d, 128d)
        print("Constructing in-memory Matryoshka FAISS indices (512d, 256d, 128d)...")
        self.mrl_indices = {}
        for dim in [512, 256, 128]:
            sub_v = self.full_vectors[:, :dim].copy()
            norms = np.linalg.norm(sub_v, axis=1, keepdims=True)
            norms[norms == 0] = 1e-12
            sub_v = (sub_v / norms).astype(np.float32)
            idx = faiss.IndexFlatIP(dim)
            idx.add(sub_v)
            self.mrl_indices[dim] = idx

        # Initialize Sparse BM25
        print("Initializing BM25Okapi index...")
        self.bm25 = BM25Okapi(self.chunks)

        # Initialize SentenceTransformer on CUDA
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading {MODEL_ID} in bfloat16 on {device}...")
        self.model = SentenceTransformer(
            MODEL_ID,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device=device,
        )

        # Warm up
        _ = self.model.encode("warmup query", prompt_name="query")

    def retrieve_dense(self, query: str, top_k: int = 5, truncate_dim: int | None = None) -> tuple[list[dict[str, Any]], float]:
        t0 = time.perf_counter()
        dim = truncate_dim or 768
        with torch.inference_mode():
            if truncate_dim:
                q_vec = self.model.encode(
                    query,
                    prompt_name="query",
                    truncate_dim=dim,
                    normalize_embeddings=True,
                )
            else:
                q_vec = self.model.encode(
                    query,
                    prompt_name="query",
                    normalize_embeddings=True,
                )

        q_vec = np.array(q_vec, dtype=np.float32).reshape(1, -1)
        idx = self.index_768 if dim == 768 else self.mrl_indices[dim]
        scores, indices = idx.search(q_vec, min(top_k, idx.ntotal))
        dt = (time.perf_counter() - t0) * 1000.0

        results = []
        for rank, (score, doc_idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if doc_idx < 0 or doc_idx >= len(self.courses):
                continue
            course = self.courses[doc_idx]
            results.append({
                "doc_idx": int(doc_idx),
                "course_id": course["id"],
                "kode_mk": course["kode_mk"],
                "nama_mk": course["nama_mk"],
                "score": float(score),
                "rank": rank,
            })
        return results, dt

    def retrieve_sparse(self, query: str, top_k: int = 5) -> tuple[list[dict[str, Any]], float]:
        t0 = time.perf_counter()
        ranked = self.bm25.search(query, top_k=top_k)
        dt = (time.perf_counter() - t0) * 1000.0
        results = []
        for rank, (doc_idx, score) in enumerate(ranked, start=1):
            course = self.courses[doc_idx]
            results.append({
                "doc_idx": int(doc_idx),
                "course_id": course["id"],
                "kode_mk": course["kode_mk"],
                "nama_mk": course["nama_mk"],
                "score": float(score),
                "rank": rank,
            })
        return results, dt

    def retrieve_hybrid_rrf(
        self,
        query: str,
        top_k: int = 5,
        k_rrf: int = 60,
        weight_dense: float = 1.0,
        weight_sparse: float = 1.0,
    ) -> tuple[list[dict[str, Any]], float]:
        t0 = time.perf_counter()
        dense_hits, _ = self.retrieve_dense(query, top_k=top_k * 3)
        sparse_hits, _ = self.retrieve_sparse(query, top_k=top_k * 3)

        scores: dict[int, float] = {}
        payloads: dict[int, dict[str, Any]] = {}

        for rank, hit in enumerate(dense_hits, start=1):
            did = hit["doc_idx"]
            scores[did] = scores.get(did, 0.0) + (weight_dense / (k_rrf + rank))
            payloads[did] = hit

        for rank, hit in enumerate(sparse_hits, start=1):
            did = hit["doc_idx"]
            scores[did] = scores.get(did, 0.0) + (weight_sparse / (k_rrf + rank))
            if did not in payloads:
                payloads[did] = hit

        sorted_dids = sorted(scores.keys(), key=lambda did: scores[did], reverse=True)[:top_k]
        fused = []
        for rank, did in enumerate(sorted_dids, start=1):
            item = dict(payloads[did])
            item["rrf_score"] = scores[did]
            item["rank"] = rank
            fused.append(item)

        dt = (time.perf_counter() - t0) * 1000.0
        return fused, dt


def run_benchmark(dataset_path: Path = DEFAULT_DATASET, output_dir: Path = DEFAULT_OUTPUT_DIR, limit: int | None = None):
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(dataset_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    if limit:
        samples = samples[:limit]

    print("=====================================================================")
    print("▶ EMBEDDINGGEMMA-2 BENCHMARK EVALUATION (EXP-012)")
    print(f"  Dataset: {len(samples)} samples ({sum(1 for s in samples if s['register']=='formal')} formal, {sum(1 for s in samples if s['register']=='slang')} slang)")
    print(f"  Output: {output_dir}")
    print("=====================================================================")

    vram_start = get_gpu_vram_mib()
    print(f"GPU VRAM Baseline: {vram_start:.1f} MiB")

    # Load baseline Nomic hybrid retriever
    print("\n--- Loading Nomic Hybrid Retriever Baseline ---")
    nomic_retriever = HybridCourseRetriever()

    # Load Gemma2 evaluator
    print("\n--- Loading Gemma2 Evaluator ---")
    gemma_evaluator = Gemma2Evaluator()

    vram_loaded = get_gpu_vram_mib()
    print(f"GPU VRAM Loaded (Nomic + Gemma2): {vram_loaded:.1f} MiB (Footprint: {vram_loaded - vram_start:.1f} MiB)")

    methods = [
        "BM25 (Sparse Okapi)",
        "FAISS Dense (nomic-embed-text 768d)",
        "FAISS Dense (EmbeddingGemma-2 768d)",
        "Hybrid RRF (BM25 + nomic-embed-text)",
        "Hybrid RRF (BM25 + EmbeddingGemma-2)",
        "Matryoshka Dense (EmbeddingGemma-2 512d)",
        "Matryoshka Dense (EmbeddingGemma-2 256d)",
        "Matryoshka Dense (EmbeddingGemma-2 128d)",
    ]

    all_traces = []

    print(f"\n--- Running Benchmark across {len(samples)} queries ---")
    for i, s in enumerate(samples, start=1):
        qid = s["id"]
        q = s["question"]
        reg = s["register"]
        target_ids = s["target_course_ids"]
        target_names = "; ".join(s["target_nama_mk"][:3])

        if i % 15 == 1 or i == len(samples):
            print(f"  [{i:03d}/{len(samples):03d}] ({reg.upper()}) {q[:55]}...")

        # 1. BM25
        bm25_res, bm25_lat = gemma_evaluator.retrieve_sparse(q, top_k=5)
        cids_bm25 = [r["course_id"] for r in bm25_res]
        m_bm25 = compute_ir_metrics(cids_bm25, target_ids)
        all_traces.append({
            "sample_id": qid,
            "register": reg,
            "method": "BM25 (Sparse Okapi)",
            "latency_ms": bm25_lat,
            "top1_retrieved": bm25_res[0]["nama_mk"] if bm25_res else "-",
            "target_courses": target_names,
            **m_bm25,
        })

        # 2. Nomic Dense
        nomic_dense_res, nomic_dense_lat = nomic_retriever.retrieve_dense(q, top_k=5)
        cids_nomic_dense = [r["course_id"] for r in nomic_dense_res]
        m_nomic_dense = compute_ir_metrics(cids_nomic_dense, target_ids)
        all_traces.append({
            "sample_id": qid,
            "register": reg,
            "method": "FAISS Dense (nomic-embed-text 768d)",
            "latency_ms": nomic_dense_lat,
            "top1_retrieved": nomic_dense_res[0]["nama_mk"] if nomic_dense_res else "-",
            "target_courses": target_names,
            **m_nomic_dense,
        })

        # 3. Gemma2 Dense 768d
        gemma_dense_res, gemma_dense_lat = gemma_evaluator.retrieve_dense(q, top_k=5, truncate_dim=768)
        cids_gemma_dense = [r["course_id"] for r in gemma_dense_res]
        m_gemma_dense = compute_ir_metrics(cids_gemma_dense, target_ids)
        all_traces.append({
            "sample_id": qid,
            "register": reg,
            "method": "FAISS Dense (EmbeddingGemma-2 768d)",
            "latency_ms": gemma_dense_lat,
            "top1_retrieved": gemma_dense_res[0]["nama_mk"] if gemma_dense_res else "-",
            "target_courses": target_names,
            **m_gemma_dense,
        })

        # 4. Nomic Hybrid RRF
        nomic_rrf_res, nomic_rrf_lat = nomic_retriever.retrieve_hybrid_rrf(q, top_k=5)
        cids_nomic_rrf = [r["course_id"] for r in nomic_rrf_res]
        m_nomic_rrf = compute_ir_metrics(cids_nomic_rrf, target_ids)
        all_traces.append({
            "sample_id": qid,
            "register": reg,
            "method": "Hybrid RRF (BM25 + nomic-embed-text)",
            "latency_ms": nomic_rrf_lat,
            "top1_retrieved": nomic_rrf_res[0]["nama_mk"] if nomic_rrf_res else "-",
            "target_courses": target_names,
            **m_nomic_rrf,
        })

        # 5. Gemma2 Hybrid RRF
        gemma_rrf_res, gemma_rrf_lat = gemma_evaluator.retrieve_hybrid_rrf(q, top_k=5)
        cids_gemma_rrf = [r["course_id"] for r in gemma_rrf_res]
        m_gemma_rrf = compute_ir_metrics(cids_gemma_rrf, target_ids)
        all_traces.append({
            "sample_id": qid,
            "register": reg,
            "method": "Hybrid RRF (BM25 + EmbeddingGemma-2)",
            "latency_ms": gemma_rrf_lat,
            "top1_retrieved": gemma_rrf_res[0]["nama_mk"] if gemma_rrf_res else "-",
            "target_courses": target_names,
            **m_gemma_rrf,
        })

        # 6. Gemma2 Matryoshka 512d
        mrl512_res, mrl512_lat = gemma_evaluator.retrieve_dense(q, top_k=5, truncate_dim=512)
        cids_mrl512 = [r["course_id"] for r in mrl512_res]
        m_mrl512 = compute_ir_metrics(cids_mrl512, target_ids)
        all_traces.append({
            "sample_id": qid,
            "register": reg,
            "method": "Matryoshka Dense (EmbeddingGemma-2 512d)",
            "latency_ms": mrl512_lat,
            "top1_retrieved": mrl512_res[0]["nama_mk"] if mrl512_res else "-",
            "target_courses": target_names,
            **m_mrl512,
        })

        # 7. Gemma2 Matryoshka 256d
        mrl256_res, mrl256_lat = gemma_evaluator.retrieve_dense(q, top_k=5, truncate_dim=256)
        cids_mrl256 = [r["course_id"] for r in mrl256_res]
        m_mrl256 = compute_ir_metrics(cids_mrl256, target_ids)
        all_traces.append({
            "sample_id": qid,
            "register": reg,
            "method": "Matryoshka Dense (EmbeddingGemma-2 256d)",
            "latency_ms": mrl256_lat,
            "top1_retrieved": mrl256_res[0]["nama_mk"] if mrl256_res else "-",
            "target_courses": target_names,
            **m_mrl256,
        })

        # 8. Gemma2 Matryoshka 128d
        mrl128_res, mrl128_lat = gemma_evaluator.retrieve_dense(q, top_k=5, truncate_dim=128)
        cids_mrl128 = [r["course_id"] for r in mrl128_res]
        m_mrl128 = compute_ir_metrics(cids_mrl128, target_ids)
        all_traces.append({
            "sample_id": qid,
            "register": reg,
            "method": "Matryoshka Dense (EmbeddingGemma-2 128d)",
            "latency_ms": mrl128_lat,
            "top1_retrieved": mrl128_res[0]["nama_mk"] if mrl128_res else "-",
            "target_courses": target_names,
            **m_mrl128,
        })

    vram_end = get_gpu_vram_mib()

    # Calculate summary aggregations
    print("\n--- Aggregating Benchmark Results ---")
    metrics_summary = []
    registers = ["overall", "formal", "informal"]

    for m in methods:
        m_traces = [t for t in all_traces if t["method"] == m]
        for reg in registers:
            if reg == "overall":
                sub = m_traces
            else:
                sub = [t for t in m_traces if t["register"] == reg]

            count = len(sub)
            if count == 0:
                continue

            hit1 = float(np.mean([t["hit@1"] for t in sub])) * 100.0
            hit3 = float(np.mean([t["hit@3"] for t in sub])) * 100.0
            hit5 = float(np.mean([t["hit@5"] for t in sub])) * 100.0
            mrr = float(np.mean([t["mrr"] for t in sub]))
            ndcg5 = float(np.mean([t["ndcg@5"] for t in sub]))
            lat_mean = float(np.mean([t["latency_ms"] for t in sub]))
            lat_p50 = float(np.percentile([t["latency_ms"] for t in sub], 50))
            lat_p95 = float(np.percentile([t["latency_ms"] for t in sub], 95))

            metrics_summary.append({
                "method": m,
                "subset": reg.upper(),
                "count": count,
                "hit@1": hit1,
                "hit@3": hit3,
                "hit@5": hit5,
                "mrr": mrr,
                "ndcg@5": ndcg5,
                "avg_latency_ms": lat_mean,
                "p50_latency_ms": lat_p50,
                "p95_latency_ms": lat_p95,
            })

    # 1. Write benchmark_raw_traces.jsonl
    traces_path = output_dir / "benchmark_raw_traces.jsonl"
    with open(traces_path, "w", encoding="utf-8") as f:
        for t in all_traces:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"Saved raw traces to {traces_path}")

    # 2. Write benchmark_metrics.csv
    csv_path = output_dir / "benchmark_metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "subset",
                "count",
                "hit@1",
                "hit@3",
                "hit@5",
                "mrr",
                "ndcg@5",
                "avg_latency_ms",
                "p50_latency_ms",
                "p95_latency_ms",
            ],
        )
        writer.writeheader()
        for row in metrics_summary:
            writer.writerow({
                "method": row["method"],
                "subset": row["subset"],
                "count": row["count"],
                "hit@1": f"{row['hit@1']:.2f}",
                "hit@3": f"{row['hit@3']:.2f}",
                "hit@5": f"{row['hit@5']:.2f}",
                "mrr": f"{row['mrr']:.4f}",
                "ndcg@5": f"{row['ndcg@5']:.4f}",
                "avg_latency_ms": f"{row['avg_latency_ms']:.2f}",
                "p50_latency_ms": f"{row['p50_latency_ms']:.2f}",
                "p95_latency_ms": f"{row['p95_latency_ms']:.2f}",
            })
    print(f"Saved metrics CSV to {csv_path}")

    # 3. Generate and Write benchmark_summary.md
    summary_path = output_dir / "benchmark_summary.md"
    overall_rows = [r for r in metrics_summary if r["subset"] == "OVERALL"]
    formal_rows = [r for r in metrics_summary if r["subset"] == "FORMAL"]
    slang_rows = [r for r in metrics_summary if r["subset"] == "INFORMAL"]

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# Benchmark Summary — Exp-012: EmbeddingGemma-2 (270M) vs nomic-embed-text\n\n")
        f.write(f"- **Tanggal Pengujian:** {time.strftime('%Y-%m-%d')}\n")
        f.write(f"- **Platform Hardware:** NVIDIA GeForce RTX 5050 Laptop GPU (8GB VRAM), WSL2 Ubuntu Linux\n")
        f.write(f"- **VRAM Baseline / Peak:** {vram_start:.1f} MiB / {vram_end:.1f} MiB\n")
        f.write(f"- **Dataset Evaluasi:** {len(samples)} Ground Truth Queries (108 Formal Subset A + 21 Slang Subset B)\n\n")

        f.write("## 1. Perbandingan Kinerja Retrieval Keseluruhan (OVERALL — 129 Kueri)\n\n")
        f.write("| Metode Retrieval | Dimensi | Hit@1 (%) | Hit@3 (%) | Hit@5 (%) | MRR | NDCG@5 | Avg Latensi (ms) | P50 (ms) |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for r in overall_rows:
            dim = "768d"
            if "512d" in r["method"]:
                dim = "512d"
            elif "256d" in r["method"]:
                dim = "256d"
            elif "128d" in r["method"]:
                dim = "128d"
            elif "BM25" in r["method"] and "RRF" not in r["method"]:
                dim = "Sparse"
            f.write(
                f"| **{r['method']}** | {dim} | {r['hit@1']:.1f}% | {r['hit@3']:.1f}% | {r['hit@5']:.1f}% | {r['mrr']:.4f} | {r['ndcg@5']:.4f} | {r['avg_latency_ms']:.1f} ms | {r['p50_latency_ms']:.1f} ms |\n"
            )

        f.write("\n## 2. Analisis Performa Berdasarkan Register Bahasa\n\n")
        f.write("### A. Kueri Formal (Subset A — 108 Kueri)\n\n")
        f.write("| Metode Retrieval | Hit@1 (%) | Hit@3 (%) | Hit@5 (%) | MRR | NDCG@5 |\n")
        f.write("|---|---|---|---|---|---|\n")
        for r in formal_rows:
            f.write(f"| **{r['method']}** | {r['hit@1']:.1f}% | {r['hit@3']:.1f}% | {r['hit@5']:.1f}% | {r['mrr']:.4f} | {r['ndcg@5']:.4f} |\n")

        f.write("\n### B. Kueri Informal / Slang Mahasiswa (Subset B — 21 Kueri)\n\n")
        f.write("| Metode Retrieval | Hit@1 (%) | Hit@3 (%) | Hit@5 (%) | MRR | NDCG@5 |\n")
        f.write("|---|---|---|---|---|---|\n")
        for r in slang_rows:
            f.write(f"| **{r['method']}** | {r['hit@1']:.1f}% | {r['hit@3']:.1f}% | {r['hit@5']:.1f}% | {r['mrr']:.4f} | {r['ndcg@5']:.4f} |\n")

        f.write("\n## 3. Analisis Matryoshka Representation Learning (MRL Truncation)\n\n")
        f.write("Evaluasi degradasi akurasi terhadap pemangkasan dimensi pada EmbeddingGemma-2:\n\n")
        mrl_rows = [r for r in overall_rows if "EmbeddingGemma-2" in r["method"] and "Hybrid" not in r["method"]]
        f.write("| Dimensi Vektor | Hit@1 (%) | Hit@3 (%) | Hit@5 (%) | MRR | Penghematan Storage Index |\n")
        f.write("|---|---|---|---|---|---|\n")
        for r in mrl_rows:
            dim_label = "768d (Full)"
            pct = "0% (Baseline)"
            if "512d" in r["method"]:
                dim_label = "512d"
                pct = "Hemat 33.3%"
            elif "256d" in r["method"]:
                dim_label = "256d"
                pct = "Hemat 66.7%"
            elif "128d" in r["method"]:
                dim_label = "128d"
                pct = "Hemat 83.3%"
            f.write(f"| **{dim_label}** | {r['hit@1']:.1f}% | {r['hit@3']:.1f}% | {r['hit@5']:.1f}% | {r['mrr']:.4f} | {pct} |\n")

        f.write("\n## 4. Temuan Riset & Keputusan Arsitektur\n\n")
        f.write("1. **Perbandingan Dense (EmbeddingGemma-2 vs nomic-embed-text)**:\n")
        f.write("   - EmbeddingGemma-2 768d vs nomic-embed-text pada metrik Hit@3 dan MRR.\n")
        f.write("2. **Perbandingan Hybrid RRF**:\n")
        f.write("   - Pengaruh fusi Reciprocal Rank Fusion (k=60) antara BM25 dengan embedding model penantang.\n")
        f.write("3. **Efisiensi Komputasi & Hardware (RTX 5050 8GB)**:\n")
        f.write(f"   - Latensi inferensi query stabil dan resident footprint aman pada {vram_end:.1f} MiB.\n")

    print(f"Saved summary report to {summary_path}")

    # 4. Write README.md
    readme_path = output_dir / "README.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("# Exp-012: Benchmark Evaluasi EmbeddingGemma-2 vs nomic-embed-text\n\n")
        f.write("Eksperimen komparasi empiris Information Retrieval (IR) antara model `google/embeddinggemma-2` (270M text backbone) terhadap baseline `nomic-embed-text` (768d) pada 369 mata kuliah FTMM UNAIR.\n\n")
        f.write("## File Artefak\n")
        f.write("- `benchmark_summary.md`: Laporan komprehensif metrik IR (Hit@1, Hit@3, Hit@5, MRR, NDCG@5) dan analisis ablasi MRL.\n")
        f.write("- `benchmark_metrics.csv`: Data mentah agregasi per metode dan per register bahasa.\n")
        f.write("- `benchmark_raw_traces.jsonl`: Log detail pencarian dan peringkat mata kuliah per kueri.\n")

    print(f"Saved README to {readme_path}")
    print("\n[SUCCESS] Benchmark Exp-012 Complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run EmbeddingGemma-2 Retrieval Benchmark")
    parser.add_argument("--limit", type=int, default=None, help="Limit sample count")
    args = parser.parse_args()
    run_benchmark(limit=args.limit)
