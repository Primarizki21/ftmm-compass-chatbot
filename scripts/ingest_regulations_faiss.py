"""
Skrip kompilasi indeks FAISS untuk Regulasi Akademik & SOP Institusi FTMM UNAIR.
Mengindeks:
1. Panduan Skripsi & Tugas Akhir (152 butir)
2. Panduan Praktik Kerja Lapangan / PKL (45 butir)
3. Panduan Yudisium & Kelulusan (24 butir)
4. Panduan Akademik Fakultas (33 butir)
5. SOP Pengajuan Layanan Administrasi (8 SOP)
Total: 262 butir pengetahuan regulasi berbutir halus.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import faiss
import numpy as np

# WSL2 threading & environment
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.engine.base import get_ollama_client

DATA_DIR = Path("data/ground_truth")
OUTPUT_DIR = Path("data/faiss")
INDEX_PATH = OUTPUT_DIR / "ftmm_regulations.faiss"
META_PATH = OUTPUT_DIR / "ftmm_regulations_meta.json"

EMBED_MODEL = os.getenv("EMBED_MODEL_NAME", "nomic-embed-text-v2-moe")
EMBED_DIM = int(os.getenv("EMBED_DIMENSION", "768"))
BATCH_SIZE = 32

REGULATION_SOURCES = [
    {
        "file": "panduanskripsi.json",
        "topic": "skripsi",
        "default_source": "Panduan Skripsi FTMM UNAIR",
    },
    {
        "file": "panduanpkl.json",
        "topic": "pkl",
        "default_source": "Panduan PKL FTMM UNAIR",
    },
    {
        "file": "panduanyudisium.json",
        "topic": "yudisium",
        "default_source": "Panduan Yudisium FTMM UNAIR",
    },
    {
        "file": "panduanakademik.json",
        "topic": "akademik",
        "default_source": "Panduan Akademik FTMM UNAIR",
    },
    {
        "file": "google_drive_sop.json",
        "topic": "sop",
        "default_source": "SOP Administrasi Akademik FTMM UNAIR",
    },
]


def format_regulation_chunk(item: dict[str, Any], default_source: str, topic: str) -> str:
    source = item.get("source_name") or default_source
    chapter = item.get("chapter") or item.get("category") or ""
    subchapter = item.get("subchapter", "")
    title = item.get("title", "").strip()
    desc = item.get("description", "").strip()
    steps = item.get("step_by_step", "").strip()

    parts = [f"[REGULASI: {source.upper()}]", f"Kategori/Topik: {topic.upper()}"]
    if chapter:
        parts.append(f"Bab: {chapter}")
    if subchapter:
        parts.append(f"Pasal/Sub-Bab: {subchapter}")
    if title:
        parts.append(f"Judul: {title}")
    if desc:
        parts.append(f"Ketentuan/Isi: {desc}")
    if steps:
        parts.append(f"Prosedur/Alur: {steps}")
    return "\n".join(parts)


def load_all_regulations() -> tuple[list[dict[str, Any]], list[str]]:
    all_items: list[dict[str, Any]] = []
    all_chunks: list[str] = []

    for cfg in REGULATION_SOURCES:
        fpath = DATA_DIR / cfg["file"]
        if not fpath.exists():
            print(f"Warning: {fpath} tidak ditemukan, dilewati.")
            continue

        with open(fpath, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        if not isinstance(raw_data, list):
            print(f"Warning: format data di {fpath} bukan list, dilewati.")
            continue

        topic = cfg["topic"]
        source_name = cfg["default_source"]

        for idx, item in enumerate(raw_data):
            chunk = format_regulation_chunk(item, source_name, topic)
            item_id = str(item.get("id") or f"{topic}_{idx + 1}")
            entry = {
                "id": f"{topic}_{item_id}",
                "topic": topic,
                "source_name": item.get("source_name") or source_name,
                "chapter": item.get("chapter") or item.get("category") or "",
                "subchapter": item.get("subchapter", ""),
                "title": item.get("title", ""),
                "description": item.get("description", ""),
                "step_by_step": item.get("step_by_step", ""),
                "url": item.get("source_url", ""),
                "doc_idx": len(all_items),
            }
            all_items.append(entry)
            all_chunks.append(chunk)

    print(f"Total butir regulasi termuat: {len(all_items)} butir dari {len(REGULATION_SOURCES)} berkas.")
    return all_items, all_chunks


def generate_embeddings(chunks: list[str], client: Any) -> np.ndarray:
    print(f"Generating embeddings for {len(chunks)} regulation chunks using {EMBED_MODEL}...")
    all_embeddings = []
    t0 = time.perf_counter()

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        resp = client.embed(model=EMBED_MODEL, input=batch, keep_alive=-1)
        batch_vecs = np.array(resp["embeddings"], dtype=np.float32)
        norms = np.linalg.norm(batch_vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        batch_vecs = batch_vecs / norms
        all_embeddings.append(batch_vecs)
        print(f"  Processed batch {i // BATCH_SIZE + 1}/{(len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE} ({len(batch)} chunks)")

    emb_matrix = np.vstack(all_embeddings)
    dt = time.perf_counter() - t0
    print(f"Embeddings generated in {dt:.2f}s ({dt / len(chunks) * 1000:.2f} ms/chunk). Shape: {emb_matrix.shape}")
    return emb_matrix


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    items, chunks = load_all_regulations()

    if not items:
        print("Error: Tidak ada data regulasi yang berhasil dimuat.")
        sys.exit(1)

    client = get_ollama_client()
    embeddings = generate_embeddings(chunks, client)

    # Build FAISS IndexFlatIP
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(embeddings)
    print(f"Indexed {index.ntotal} vectors to FAISS IndexFlatIP ({EMBED_DIM}d).")

    faiss.write_index(index, str(INDEX_PATH))

    # Save metadata
    metadata = {
        "regulations": items,
        "chunks": chunks,
        "embed_model": EMBED_MODEL,
        "dim": EMBED_DIM,
        "total": len(items),
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"\n[SUCCESS] Index regulasi tersimpan di {INDEX_PATH}")
    print(f"[SUCCESS] Metadata regulasi tersimpan di {META_PATH} ({len(items)} items)")


if __name__ == "__main__":
    main()
