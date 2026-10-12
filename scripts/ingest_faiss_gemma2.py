import json
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pickle
import time
from pathlib import Path
import faiss
import numpy as np
import torch
torch.set_num_threads(4)
from sentence_transformers import SentenceTransformer
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.ingest_faiss import format_course_chunk, load_all_courses
OUTPUT_DIR = Path("data/faiss")
INDEX_PATH = OUTPUT_DIR / "ftmm_courses_gemma2.faiss"
META_PATH = OUTPUT_DIR / "ftmm_courses_gemma2_meta.pkl"
INDEX_256D_PATH = OUTPUT_DIR / "ftmm_courses_gemma2_256d.faiss"
MODEL_ID = "google/embeddinggemma-2"
EMBED_DIM = 768
MRL_DIM = 256
BATCH_SIZE = 16


def generate_gemma2_embeddings(texts: list[str], model: SentenceTransformer) -> np.ndarray:
    print(f"Generating embeddings for {len(texts)} courses using {MODEL_ID} (prompt_name='document')...")
    t0 = time.perf_counter()
    with torch.inference_mode():
        embeddings = model.encode(
            texts,
            batch_size=BATCH_SIZE,
            prompt_name="document",
            normalize_embeddings=True,
            show_progress_bar=True,
        )
    dt = time.perf_counter() - t0
    print(f"Embeddings generated in {dt:.2f}s ({dt / len(texts) * 1000:.2f} ms/course).")
    return np.array(embeddings, dtype=np.float32)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    courses = load_all_courses()
    chunks = [format_course_chunk(mk) for mk in courses]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {MODEL_ID} on {device} (bfloat16)...")
    model = SentenceTransformer(
        MODEL_ID,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device=device,
    )

    embeddings = generate_gemma2_embeddings(chunks, model)
    print(f"Embedding matrix shape: {embeddings.shape}")

    # Build FAISS IndexFlatIP (Cosine similarity since embeddings are L2 normalized)
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(embeddings)
    print(f"Indexed {index.ntotal} vectors to FAISS IndexFlatIP ({EMBED_DIM}d).")

    faiss.write_index(index, str(INDEX_PATH))

    # Build FAISS IndexFlatIP for MRL 256d
    embeddings_256d = embeddings[:, :MRL_DIM].copy()
    norms = np.linalg.norm(embeddings_256d, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    embeddings_256d = embeddings_256d / norms

    index_256d = faiss.IndexFlatIP(MRL_DIM)
    index_256d.add(embeddings_256d)
    faiss.write_index(index_256d, str(INDEX_256D_PATH))
    print(f"Indexed {index_256d.ntotal} vectors to FAISS IndexFlatIP MRL ({MRL_DIM}d).")
    print(f"[SUCCESS] Index 256d saved to {INDEX_256D_PATH}")
    metadata = {
        "courses": courses,
        "chunks": chunks,
        "embed_model": MODEL_ID,
        "dim": EMBED_DIM,
        "total": len(courses),
    }
    with open(META_PATH, "wb") as f:
        pickle.dump(metadata, f)

    print(f"\n[SUCCESS] Index saved to {INDEX_PATH}")
    print(f"[SUCCESS] Metadata saved to {META_PATH}")


if __name__ == "__main__":
    main()
