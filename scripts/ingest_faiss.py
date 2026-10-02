"""
Script Ingestion FAISS untuk Katalog Kurikulum 5 Prodi FTMM UNAIR.
Menerapkan teknik Row-based & Column-aware Chunking (TaBERT & RAT-SQL principle).
Menggunakan Ollama embedding nomic-embed-text (768 dimensi, Cosine IndexFlatIP).
"""

import json
import os
import pickle
from pathlib import Path
import numpy as np
import faiss
import ollama

EKSTRAK_DIR = Path("Ekstrak")
OUTPUT_DIR = Path("data/faiss")
INDEX_PATH = OUTPUT_DIR / "ftmm_courses.faiss"
META_PATH = OUTPUT_DIR / "ftmm_courses_meta.pkl"
EMBED_MODEL = "nomic-embed-text"
EMBED_DIM = 768
BATCH_SIZE = 32


def format_course_chunk(mk: dict) -> str:
    """
    Format baris mata kuliah menjadi chunk kolom eksplisit (Column-aware).
    Memastikan representasi teks mempertahankan asosiasi atribut penting.
    """
    prodi = mk.get("program_studi", "FTMM")
    kode = mk.get("kode_mk", "").strip()
    nama = mk.get("nama_mk", "").strip()
    sks = mk.get("sks", 0)
    sem = mk.get("semester", 0)
    jenis = mk.get("jenis", "wajib")
    
    prasyarat_list = mk.get("prasyarat", [])
    if isinstance(prasyarat_list, list) and prasyarat_list:
        prasyarat = ", ".join(str(p).strip() for p in prasyarat_list)
    else:
        prasyarat = "Tidak Ada"
        
    deskripsi = mk.get("deskripsi", "").strip()
    
    cp_list = mk.get("capaian_pembelajaran", [])
    if isinstance(cp_list, list):
        cp = " ".join(str(c).strip() for c in cp_list)
    else:
        cp = str(cp_list).strip()
        
    dosen_list = mk.get("dosen_pengampu", [])
    if isinstance(dosen_list, list) and dosen_list:
        dosen = ", ".join(str(d).strip() for d in dosen_list)
    else:
        dosen = "Dosen Pengampu FTMM"

    chunk = (
        f"[MATA KULIAH]\n"
        f"Program Studi: {prodi} | Jenis: {jenis}\n"
        f"Kode: {kode} | Nama: {nama} | SKS: {sks} | Semester: {sem}\n"
        f"Prasyarat: {prasyarat}\n"
        f"Deskripsi: {deskripsi}\n"
        f"Capaian Pembelajaran (CPMK): {cp}\n"
        f"Dosen: {dosen}"
    ).strip()
    return chunk


def load_all_courses():
    """Membaca dan memproses semua matkul dari 5 file prodi di folder Ekstrak/."""
    courses = []
    if not EKSTRAK_DIR.exists():
        raise FileNotFoundError(f"Folder {EKSTRAK_DIR} tidak ditemukan!")

    json_files = sorted(list(EKSTRAK_DIR.glob("*.json")))
    print(f"Ditemukan {len(json_files)} berkas kurikulum di {EKSTRAK_DIR}:")
    for jf in json_files:
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)
            prodi_name = data.get("metadata", {}).get("program_studi", jf.stem.replace("FIX ", ""))
            mk_list = data.get("mata_kuliah", [])
            for mk in mk_list:
                mk["program_studi"] = mk.get("program_studi") or prodi_name
                courses.append(mk)
        print(f"  - {jf.name}: {len(mk_list)} mata kuliah")

    print(f"Total mata kuliah dimuat: {len(courses)}")
    return courses


def generate_embeddings(texts: list[str]) -> np.ndarray:
    """Menghasilkan embedding via Ollama dalam batch dan menormalisasi L2."""
    all_embeddings = []
    total = len(texts)
    
    for i in range(0, total, BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        print(f"  Proses embedding batch {i // BATCH_SIZE + 1} / {int(np.ceil(total / BATCH_SIZE))} ({len(batch)} matkul)...")
        resp = ollama.embed(model=EMBED_MODEL, input=batch)
        batch_vecs = resp.get("embeddings", [])
        all_embeddings.extend(batch_vecs)
        
    vecs = np.array(all_embeddings, dtype=np.float32)
    # L2 normalize agar inner product = Cosine Similarity
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    vecs_norm = vecs / norms
    return vecs_norm


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    courses = load_all_courses()
    
    chunks = [format_course_chunk(mk) for mk in courses]
    print(f"\nContoh representasi chunk pertama:\n{'-'*50}\n{chunks[0]}\n{'-'*50}\n")
    
    print("Menghasilkan vector embeddings...")
    embeddings = generate_embeddings(chunks)
    print(f"Bentuk matriks embedding: {embeddings.shape}")
    
    # Buat FAISS IndexFlatIP (Cosine Similarity)
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(embeddings)
    print(f"Total vektor diindeks ke FAISS: {index.ntotal}")
    
    # Simpan index dan metadata
    faiss.write_index(index, str(INDEX_PATH))
    metadata = {
        "courses": courses,
        "chunks": chunks,
        "embed_model": EMBED_MODEL,
        "dim": EMBED_DIM,
        "total": len(courses)
    }
    with open(META_PATH, "wb") as f:
        pickle.dump(metadata, f)
        
    print(f"\n[SUKSES] Index tersimpan di {INDEX_PATH}")
    print(f"[SUKSES] Metadata tersimpan di {META_PATH}")


if __name__ == "__main__":
    main()
