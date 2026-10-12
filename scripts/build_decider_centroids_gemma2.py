import os

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import time
import numpy as np
import torch

torch.set_num_threads(4)
from sentence_transformers import SentenceTransformer

from src.engine.decider import DecisionCategory, PROTOTYPES

OUTPUT_PATH = Path("data/decider_centroids_gemma2.npz")
MODEL_ID = "google/embeddinggemma-2"


def build_centroids(
    model_id: str = MODEL_ID,
    output_path: Path = OUTPUT_PATH,
) -> np.ndarray:
    print(f"Loading {model_id} (bfloat16)...")
    t_load = time.perf_counter()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(
        model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device=device,
    )
    load_dt = time.perf_counter() - t_load
    print(f"Model loaded on {device} in {load_dt:.2f}s.")

    categories = [
        DecisionCategory.CHAT_ONLY,
        DecisionCategory.PREREQUISITE,
        DecisionCategory.SKS_CALCULATION,
        DecisionCategory.DEGREE_PLANNING,
        DecisionCategory.COURSE_DISCOVERY,
    ]

    centroids = []
    print("\nComputing class centroids using prompt_name='Classification'...")
    t0 = time.perf_counter()
    with torch.inference_mode():
        for cat in categories:
            texts = PROTOTYPES[cat]
            print(f"  Encoding {len(texts):2d} prototypes for [{cat.value}] ... ", end="", flush=True)
            tc0 = time.perf_counter()
            vecs = model.encode(
                texts,
                prompt_name="Classification",
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            vecs = np.array(vecs, dtype=np.float32)
            # L2 normalize just to be certain
            vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

            centroid = np.mean(vecs, axis=0)
            c_norm = np.linalg.norm(centroid)
            if c_norm > 0:
                centroid = centroid / c_norm
            centroids.append(centroid)
            tc_dt = (time.perf_counter() - tc0) * 1000
            print(f"done ({tc_dt:.1f}ms). Norm: {np.linalg.norm(centroid):.4f}")

    total_dt = (time.perf_counter() - t0) * 1000
    centroid_matrix = np.array(centroids, dtype=np.float32)
    print(f"\nAll centroids computed in {total_dt:.1f}ms. Shape: {centroid_matrix.shape}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    category_names = [cat.value for cat in categories]
    np.savez_compressed(
        output_path,
        centroids=centroid_matrix,
        categories=np.array(category_names),
        model_id=model_id,
        dim=768,
    )
    print(f"Centroids saved to {output_path} ({output_path.stat().st_size} bytes)")

    # Separability Analysis (Centroid Pairwise Cosine Similarity)
    print("\n--- Pairwise Cosine Similarity Matrix (Separability) ---")
    sim_matrix = np.dot(centroid_matrix, centroid_matrix.T)
    header = f"{'':<18}" + "".join(f"{c[:8]:>10}" for c in category_names)
    print(header)
    for i, row_cat in enumerate(category_names):
        row_str = f"{row_cat:<18}"
        for j in range(len(category_names)):
            val = sim_matrix[i, j]
            row_str += f"{val:10.4f}"
        print(row_str)

    off_diag = sim_matrix[~np.eye(sim_matrix.shape[0], dtype=bool)]
    print(f"\nMax off-diagonal similarity: {np.max(off_diag):.4f}")
    print(f"Mean off-diagonal similarity: {np.mean(off_diag):.4f}")
    print(f"Min off-diagonal similarity: {np.min(off_diag):.4f}")

    return centroid_matrix


if __name__ == "__main__":
    build_centroids()
