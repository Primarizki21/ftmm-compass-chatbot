"""
Script Ingestion FAISS untuk Katalog Kurikulum 5 Prodi FTMM UNAIR.
Menerapkan teknik Row-based & Column-aware Chunking (TaBERT & RAT-SQL principle).
Menggunakan Ollama embedding nomic-embed-text (768 dimensi, Cosine IndexFlatIP).
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any
import numpy as np
import faiss
import ollama

EKSTRAK_DIR = Path("Ekstrak")
DEFAULT_CATALOG_PATH = Path("data/courses_catalog.json")
OUTPUT_DIR = Path("data/faiss")
INDEX_PATH = OUTPUT_DIR / "ftmm_courses.faiss"
META_PATH = OUTPUT_DIR / "ftmm_courses_meta.pkl"
META_JSON_PATH = OUTPUT_DIR / "ftmm_courses_meta.json"
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


def validate_course_schema(mk: dict[str, Any], idx: int = 0) -> list[str]:
    """Memvalidasi kelayakan skema satu entri mata kuliah."""
    errors = []
    kode = str(mk.get("kode_mk", "")).strip()
    nama = str(mk.get("nama_mk", "")).strip()
    sks = mk.get("sks")
    sem = mk.get("semester")
    prodi = str(mk.get("program_studi", "")).strip()

    if not kode:
        errors.append(f"Index {idx}: 'kode_mk' tidak boleh kosong")
    if not nama:
        errors.append(f"Index {idx} ({kode or 'UNKNOWN'}): 'nama_mk' tidak boleh kosong")
    if not isinstance(sks, int) or sks < 1 or sks > 20:
        errors.append(f"Index {idx} ({kode}): 'sks' harus integer antara 1 dan 20 (nilai: {sks})")
    if not isinstance(sem, int) or sem < 1 or sem > 8:
        errors.append(f"Index {idx} ({kode}): 'semester' harus integer antara 1 dan 8 (nilai: {sem})")
    if not prodi:
        errors.append(f"Index {idx} ({kode}): 'program_studi' tidak boleh kosong")

    return errors


def validate_all_courses(courses: list[dict[str, Any]]) -> tuple[bool, list[str]]:
    """Memvalidasi seluruh korpus mata kuliah sebelum embedding."""
    all_errors = []
    for idx, mk in enumerate(courses):
        errs = validate_course_schema(mk, idx=idx)
        all_errors.extend(errs)
    return len(all_errors) == 0, all_errors


def load_catalog_or_ekstrak(source_path: Path | str | None = None) -> tuple[list[dict[str, Any]], str]:
    """Memuat katalog matkul dari JSON tunggal (SSOT) atau fallback ke folder Ekstrak/."""
    target_path = Path(source_path) if source_path else DEFAULT_CATALOG_PATH
    if target_path.exists():
        with open(target_path, "r", encoding="utf-8") as f:
            courses = json.load(f)
        return courses, str(target_path)

    if EKSTRAK_DIR.exists():
        return load_all_courses(), str(EKSTRAK_DIR)

    raise FileNotFoundError(
        f"Katalog tidak ditemukan di {target_path} dan fallback {EKSTRAK_DIR} tidak tersedia!"
    )

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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Kompilasi dataset kurikulum JSON ke FAISS vector index dan metadata."
    )
    parser.add_argument(
        "--source",
        type=str,
        default=str(DEFAULT_CATALOG_PATH),
        help="Path ke berkas JSON katalog kurikulum (default: data/courses_catalog.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUT_DIR),
        help="Direktori output untuk FAISS index dan metadata (default: data/faiss)",
    )
    parser.add_argument(
        "--embed-model",
        type=str,
        default=EMBED_MODEL,
        help="Model embedding Ollama (default: nomic-embed-text)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Hanya jalankan validasi skema JSON tanpa memanggil Ollama/FAISS",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Lanjutkan proses meskipun ditemukan peringatan validasi",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "ftmm_courses.faiss"
    meta_pkl_path = out_dir / "ftmm_courses_meta.pkl"
    meta_json_path = out_dir / "ftmm_courses_meta.json"

    courses, source_name = load_catalog_or_ekstrak(args.source)
    print(f"[LOAD] Dimuat {len(courses)} mata kuliah dari '{source_name}'.")

    is_valid, errors = validate_all_courses(courses)
    if not is_valid:
        print(f"[ERROR] Validasi skema gagal dengan {len(errors)} kesalahan:")
        for err in errors[:10]:
            print(f"  - {err}")
        if len(errors) > 10:
            print(f"  ... dan {len(errors) - 10} kesalahan lainnya.")
        if not args.force:
            print("[ABORT] Ingestion dibatalkan demi integritas data.")
            sys.exit(1)
        print("[WARN] Melanjutkan ingestion karena flag --force aktif.")
    else:
        print(f"[VALID] Seluruh {len(courses)} entri mata kuliah lolos validasi skema 100%.")

    if args.validate_only:
        print("[SUCCESS] Mode --validate-only selesai. Tidak ada kompilasi FAISS yang dijalankan.")
        return

    chunks = [format_course_chunk(mk) for mk in courses]
    print(f"\nContoh representasi chunk pertama:\n{'-'*50}\n{chunks[0]}\n{'-'*50}\n")

    print(f"Menghasilkan vector embeddings via Ollama ({args.embed_model})...")
    embeddings = generate_embeddings(chunks)
    print(f"Bentuk matriks embedding: {embeddings.shape}")

    # Buat FAISS IndexFlatIP (Cosine Similarity)
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(embeddings)
    print(f"Total vektor diindeks ke FAISS: {index.ntotal}")

    # Simpan index binary
    faiss.write_index(index, str(index_path))
    metadata = {
        "courses": courses,
        "chunks": chunks,
        "embed_model": args.embed_model,
        "dim": EMBED_DIM,
        "total": len(courses),
    }

    # Simpan metadata ganda: .pkl dan .json
    with open(meta_pkl_path, "wb") as f:
        pickle.dump(metadata, f)
    with open(meta_json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\n[SUKSES] Index FAISS tersimpan di: {index_path}")
    print(f"[SUKSES] Metadata pickle tersimpan di: {meta_pkl_path}")
    print(f"[SUKSES] Metadata JSON tersimpan di: {meta_json_path}")


if __name__ == "__main__":
    main()
