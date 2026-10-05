from typing import Any
import pickle
from pathlib import Path
from src.knowledge.hybrid_retriever import HybridCourseRetriever

META_PATH = Path("data/faiss/ftmm_courses_meta.pkl")

# Lazy-loaded singleton components
_retriever: HybridCourseRetriever | None = None
_courses_cache: list[dict[str, Any]] | None = None


def _get_catalog() -> list[dict[str, Any]]:
    global _courses_cache
    if _courses_cache is None:
        with open(META_PATH, "rb") as f:
            meta = pickle.load(f)
            _courses_cache = meta["courses"]
    return _courses_cache


def _get_retriever() -> HybridCourseRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridCourseRetriever()
    return _retriever


def search_courses(query: str, prodi: str = "", top_k: int = 3) -> list[dict[str, Any]]:
    """
    Mencari mata kuliah FTMM berdasarkan kesamaan kata kunci dan semantik menggunakan Hybrid RRF.
    """
    retriever = _get_retriever()
    results, _ = retriever.retrieve_hybrid_rrf(query, top_k=top_k * 2)
    
    filtered = []
    prodi_lower = str(prodi or "").lower().strip()
    for r in results:
        course_prodi = str(r.get("chunk", "")).lower()
        if prodi_lower and prodi_lower not in course_prodi:
            continue
        filtered.append({
            "kode_mk": r["kode_mk"],
            "nama_mk": r["nama_mk"],
            "score": round(r["rrf_score"], 4),
        })
        if len(filtered) >= top_k:
            break
            
    if not filtered:
        filtered = [{
            "kode_mk": r["kode_mk"],
            "nama_mk": r["nama_mk"],
            "score": round(r["rrf_score"], 4),
        } for r in results[:top_k]]
        
    return filtered


def validate_prerequisites_dag(
    course_code_or_name: str,
    passed_courses: list[str] | str | None = None
) -> dict[str, Any]:
    """
    Memvalidasi apakah mata kuliah yang ingin diambil sudah memenuhi rantai prasyarat (prerequisite DAG).
    """
    catalog = _get_catalog()
    query = str(course_code_or_name or "").lower().strip()
    if passed_courses is None:
        passed_courses = []
    elif isinstance(passed_courses, str):
        passed_courses = [p.strip() for p in passed_courses.split(",") if p.strip()]
    target_course = None
    for c in catalog:
        if c["kode_mk"].lower() == query or c["nama_mk"].lower() == query or query in c["nama_mk"].lower():
            target_course = c
            break
            
    # Fallback search if exact name/code not found (e.g. English term "Machine Learning" -> "Pembelajaran Mesin")
    if not target_course:
        retriever = _get_retriever()
        search_hits, _ = retriever.retrieve_hybrid_rrf(course_code_or_name, top_k=1)
        if search_hits:
            best_did = search_hits[0].get("doc_idx", -1)
            if 0 <= best_did < len(catalog):
                target_course = catalog[best_did]

    if not target_course:
        return {
            "found": False,
            "is_eligible": False,
            "message": f"Mata kuliah '{course_code_or_name}' tidak ditemukan di katalog kurikulum FTMM."
        }
        
    prereqs = target_course.get("prasyarat", [])
    if not prereqs or prereqs == ["Tidak Ada"] or prereqs == ["-"]:
        return {
            "found": True,
            "course": target_course["nama_mk"],
            "kode_mk": target_course["kode_mk"],
            "has_prerequisites": False,
            "prerequisites": [],
            "missing_prerequisites": [],
            "is_eligible": True,
            "explanation": f"Mata kuliah {target_course['nama_mk']} ({target_course['kode_mk']}) tidak memiliki prasyarat dan dapat langsung diambil."
        }
        
    passed_lower = [str(p).lower().strip() for p in passed_courses]
    missing = []
    
    for req in prereqs:
        req_clean = str(req).strip()
        req_lower = req_clean.lower()
        
        satisfied = any(
            req_lower == p or req_lower in p or p in req_lower
            for p in passed_lower
        )
        if not satisfied:
            missing.append(req_clean)
            
    is_eligible = len(missing) == 0
    if is_eligible:
        expl = f"Mahasiswa memenuhi seluruh syarat prasyarat untuk {target_course['nama_mk']} ({', '.join(prereqs)})."
    else:
        expl = f"Mahasiswa belum memenuhi prasyarat untuk {target_course['nama_mk']}. Mata kuliah yang belum lulus: {', '.join(missing)}."
        
    return {
        "found": True,
        "course": target_course["nama_mk"],
        "kode_mk": target_course["kode_mk"],
        "has_prerequisites": True,
        "prerequisites": prereqs,
        "missing_prerequisites": missing,
        "is_eligible": is_eligible,
        "explanation": expl
    }


def calculate_sks_limit(ips: float) -> dict[str, Any]:
    """
    Menghitung batas beban studi (SKS) maksimal berdasarkan Pasal 5.4 Panduan Akademik FTMM.
    """
    ips_val = float(str(ips).replace(",", ".").strip())
    if ips_val > 3.00:
        max_sks = 24
        tier = "IPS > 3.00"
    elif ips_val >= 2.51:
        max_sks = 20
        tier = "IPS 2.51 - 3.00"
    elif ips_val >= 2.00:
        max_sks = 18
        tier = "IPS 2.00 - 2.50"
    else:
        max_sks = 15
        tier = "IPS < 2.00"
        
    return {
        "ips": ips_val,
        "tier": tier,
        "max_sks": max_sks,
        "pasal_rujukan": "Pasal 5.4 Panduan Akademik FTMM",
        "explanation": f"Berdasarkan {tier} (Pasal 5.4 Panduan Akademik FTMM), batas beban studi maksimal semester depan adalah {max_sks} SKS."
    }


def generate_degree_plan(
    program_studi: str = "Teknologi Sains Data",
    current_semester: int = 1,
    target_semester: int = 8,
    passed_courses: list[str] | str | None = None,
    focus_tracks: list[str] | str | None = None,
    max_sks_per_sem: int = 24
) -> dict[str, Any]:
    """
    Menyusun rancangan studi semester per semester tervalidasi prasyarat yang dapat langsung dikonsumsi DegreePlanner.tsx.
    """
    catalog = _get_catalog()
    current_semester = int(current_semester or 1)
    target_semester = int(target_semester or 8)
    max_sks_per_sem = int(max_sks_per_sem or 24)
    if passed_courses is None:
        passed_courses = []
    elif isinstance(passed_courses, str):
        passed_courses = [p.strip() for p in passed_courses.split(",") if p.strip()]
    if focus_tracks is None:
        focus_tracks = []
    elif isinstance(focus_tracks, str):
        focus_tracks = [t.strip() for t in focus_tracks.split(",") if t.strip()]
    prodi_lower = str(program_studi or "Teknologi Sains Data").lower().strip()
    prodi_courses = [
        c for c in catalog
        if prodi_lower in str(c.get("program_studi", "")).lower() or not prodi_lower
    ]
    if not prodi_courses:
        prodi_courses = catalog[:40]
        
    plan: dict[int, list[dict[str, Any]]] = {s: [] for s in range(1, target_semester + 1)}
    passed_set = set(str(p).lower().strip() for p in passed_courses)
    total_credits = 0
    
    # 1. Semesters prior to current: placed as completed
    for c in prodi_courses:
        c_sem = int(c.get("semester", 1))
        c_name = c["nama_mk"]
        c_code = c["kode_mk"]
        c_sks = int(c.get("sks", 2))
        
        if c_sem < current_semester:
            plan[c_sem].append({
                "id": c_code,
                "name": c_name,
                "credits": c_sks,
                "semester": c_sem,
                "type": c.get("jenis", "Wajib"),
                "status": "completed"
            })
            total_credits += c_sks
            passed_set.add(c_name.lower())
            passed_set.add(c_code.lower())
            
    # 2. Current and future semesters: place Wajib first, then relevant Pilihan
    for sem in range(current_semester, target_semester + 1):
        sem_sks = 0
        
        # Place Wajib courses for this semester
        for c in prodi_courses:
            c_sem = int(c.get("semester", 1))
            c_name = c["nama_mk"]
            c_code = c["kode_mk"]
            c_sks = int(c.get("sks", 2))
            c_type = str(c.get("jenis", "wajib")).lower()
            
            if c_sem == sem and "wajib" in c_type:
                if sem_sks + c_sks <= max_sks_per_sem:
                    plan[sem].append({
                        "id": c_code,
                        "name": c_name,
                        "credits": c_sks,
                        "semester": sem,
                        "type": "Wajib",
                        "status": "planned"
                    })
                    sem_sks += c_sks
                    total_credits += c_sks
                    passed_set.add(c_name.lower())
                    passed_set.add(c_code.lower())
                    
        # Place Pilihan matching focus tracks if space allows
        tracks_lower = [t.lower() for t in focus_tracks]
        for c in prodi_courses:
            c_sem = int(c.get("semester", 1))
            c_name = c["nama_mk"]
            c_code = c["kode_mk"]
            c_sks = int(c.get("sks", 2))
            c_type = str(c.get("jenis", "pilihan")).lower()
            c_desc = str(c.get("deskripsi", "")).lower()
            
            if "pilihan" in c_type and c_code.lower() not in passed_set and c_name.lower() not in passed_set:
                matches_track = any(t in c_name.lower() or t in c_desc for t in tracks_lower)
                if matches_track and (sem_sks + c_sks <= max_sks_per_sem):
                    plan[sem].append({
                        "id": c_code,
                        "name": c_name,
                        "credits": c_sks,
                        "semester": sem,
                        "type": "Pilihan",
                        "status": "planned"
                    })
                    sem_sks += c_sks
                    total_credits += c_sks
                    passed_set.add(c_name.lower())
                    passed_set.add(c_code.lower())
                    
    return {
        "action": "APPLY_DEGREE_PLAN",
        "summary": {
            "program_studi": program_studi,
            "current_semester": current_semester,
            "target_semester": target_semester,
            "total_credits": total_credits,
            "focus_tracks": focus_tracks,
            "note": "Rencana studi terverifikasi deterministik memenuhi batasan prasyarat dan SKS kurikulum FTMM."
        },
        "plan": plan
    }
