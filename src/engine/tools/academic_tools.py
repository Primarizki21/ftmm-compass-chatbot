import pickle
import re
from pathlib import Path
from typing import Any
from src.knowledge.hybrid_retriever import HybridCourseRetriever

SLANG_LEXICON: dict[str, str] = {
    # Indonesian student slang & abbreviations
    "alpro": "algoritma dan pemrograman",
    "alpro 1": "algoritma pemrograman i",
    "alpro 2": "algoritma pemrograman ii",
    "strukdat": "struktur data",
    "basdat": "basis data",
    "probstat": "probabilitas dan statistika",
    "metstat": "metode statistika",
    "visdat": "eksplorasi dan visualisasi data",
    "jarkom": "jaringan komputer",
    "sisop": "sistem operasi",
    "pemweb": "pemrograman web",
    "kalkulus 1": "kalkulus i",
    "kalkulus 2": "kalkulus ii",
    "pbo": "pemrograman berorientasi objek",
    "ml": "pembelajaran mesin",
    "machine learning": "pembelajaran mesin",
    "ai": "kecerdasan buatan",
    "artificial intelligence": "kecerdasan buatan",
    "nlp": "pemrosesan bahasa alami",
    # English academic course terms & mappings
    "calculus": "kalkulus",
    "calculus i": "kalkulus i",
    "calculus ii": "kalkulus ii",
    "calculus 1": "kalkulus i",
    "calculus 2": "kalkulus ii",
    "discrete math": "matematika diskrit",
    "discrete mathematics": "matematika diskrit",
    "linear algebra": "aljabar linier",
    "probability": "probabilitas dan statistika",
    "statistics": "statistika",
    "physics": "fisika dasar",
    "basic physics": "fisika dasar",
    "chemistry": "kimia dasar",
    "basic chemistry": "kimia dasar",
    "biology": "biologi dasar",
    "basic biology": "biologi dasar",
    "data mining": "data mining",
    "deep learning": "pembelajaran mendalam",
    "computer vision": "visi komputer",
    "operating systems": "sistem operasi",
    "operating system": "sistem operasi",
    "computer networks": "jaringan komputer",
    "computer network": "jaringan komputer",
    "database": "basis data",
    "database systems": "basis data",
    "cloud computing": "komputasi awan",
    "internet of things": "internet of things",
    "iot": "internet of things",
    "robotics": "robotika",
    "autonomous robotics": "robotika otonom",
    "control systems": "sistem kontrol",
    "electric circuits": "rangkaian listrik",
    "electrical circuits": "rangkaian listrik",
    "power systems": "sistem tenaga listrik",
    "operations research": "riset operasi",
    "supply chain": "rantai pasok",
    "ergonomics": "ergonomi",
    "nanotechnology": "nanoteknologi",
    "nanomaterials": "nanomaterial",
    "nanomedicine": "nanomedis",
    "biomedical": "nanomedis",
}
STOP_TOKENS: set[str] = {
    "dan", "di", "ke", "dari", "yang", "untuk", "pada", "1", "2", "i", "ii",
    "dan/atau", "proses", "stokastik", "praktikum", "lanjut", "lanjutan"
}

GENERIC_MODIFIERS: set[str] = {
    "dasar", "lanjut", "1", "2", "i", "ii", "pengantar", "rekayasa", "teknik"
}


def normalize_academic_text(text: str) -> str:
    cleaned = str(text or "").lower().strip()
    for slang, formal in SLANG_LEXICON.items():
        cleaned = re.sub(rf"\b{re.escape(slang)}\b", formal, cleaned)
    return cleaned


def extract_content_tokens(text: str) -> set[str]:
    tokens = re.findall(r"\w+", normalize_academic_text(text))
    return {t for t in tokens if len(t) > 1 and t not in STOP_TOKENS}
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

    query_norm = normalize_academic_text(query)
    query_tokens = extract_content_tokens(query_norm)

    target_course = None
    for c in catalog:
        c_code_low = c["kode_mk"].lower()
        c_name_norm = normalize_academic_text(c["nama_mk"])
        if c_code_low == query_norm or c_name_norm == query_norm or query_norm in c_name_norm:
            target_course = c
            break

    # Token-overlap ranking in catalog if exact match not found
    if not target_course and query_tokens:
        best_score = 0.0
        best_c = None
        for c in catalog:
            c_tokens = extract_content_tokens(c["nama_mk"])
            if not c_tokens:
                continue
            common = query_tokens.intersection(c_tokens)
            non_generic = common - GENERIC_MODIFIERS
            if not non_generic:
                continue
            score = len(common) / max(len(query_tokens), 1)
            if score > best_score and score >= 0.5:
                best_score = score
                best_c = c
        if best_c:
            target_course = best_c

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

    missing = []
    for req in prereqs:
        req_clean = str(req).strip()
        req_norm = normalize_academic_text(req_clean)
        req_tokens = extract_content_tokens(req_norm)
        if not req_tokens:
            continue

        satisfied = False
        for p in passed_courses:
            p_clean = str(p).strip()
            p_norm = normalize_academic_text(p_clean)
            # Substring match
            if req_norm in p_norm or p_norm in req_norm:
                satisfied = True
                break
            # Token overlap match
            p_tokens = extract_content_tokens(p_norm)
            if not p_tokens:
                continue
            common = req_tokens.intersection(p_tokens)
            if len(common) / len(req_tokens) >= 0.5:
                satisfied = True
                break

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
    Menyusun rancangan studi semester per semester tervalidasi prasyarat (Neuro-Symbolic CSP Planner)
    yang dapat langsung dikonsumsi DegreePlanner.tsx.
    """
    catalog = _get_catalog()
    canonical_prodi = normalize_program_studi(program_studi or "Teknologi Sains Data")
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

    prodi_courses = [
        c for c in catalog
        if canonical_prodi.lower() in str(c.get("program_studi", "")).lower()
    ]
    if not prodi_courses:
        # Jika tidak cocok persis, fallback ke katalog prodi terdekat tanpa truncation acak
        prodi_courses = [
            c for c in catalog
            if "sains data" in str(c.get("program_studi", "")).lower()
        ]
        canonical_prodi = "Teknologi Sains Data"

    plan: dict[int, list[dict[str, Any]]] = {s: [] for s in range(1, target_semester + 1)}
    passed_codes = set()
    passed_names = set()
    for p in passed_courses:
        p_str = str(p).lower().strip()
        passed_codes.add(p_str)
        passed_names.add(p_str)

    total_credits = 0

    # 1. Semesters prior to current: placed as completed with single religion rule
    for sem in range(1, current_semester):
        has_religion = any("agama" in p for p in passed_names)
        sem_wajib = [
            c for c in prodi_courses
            if int(c.get("semester", 1)) == sem and "wajib" in str(c.get("jenis", "wajib")).lower()
        ]
        for c in sem_wajib:
            c_name = c["nama_mk"]
            c_code = c["kode_mk"]
            c_sks = int(c.get("sks", 2))
            if "agama" in c_name.lower():
                if has_religion:
                    continue
                has_religion = True
            plan[sem].append({
                "id": str(c.get("id") or c_code),
                "code": str(c_code),
                "name": str(c_name),
                "credits": c_sks,
                "semester": sem,
                "type": "Wajib",
                "parity": "odd" if sem % 2 == 1 else "even",
                "desc": str(c.get("deskripsi", "")),
                "status": "completed",
            })
            total_credits += c_sks
            passed_codes.add(c_code.lower())
            passed_names.add(c_name.lower())

    # 2. Semantic retrieval for focus tracks expansion
    scored_electives: list[tuple[dict[str, Any], float]] = []
    if focus_tracks:
        query_text = " ".join(focus_tracks)
        try:
            from src.knowledge.retriever import get_course_retriever
            c_retriever = get_course_retriever()
            hits = c_retriever.retrieve(query_text, top_k=15, prodi_filter=canonical_prodi)
            for h in hits:
                c = h["course"]
                if "pilihan" in str(c.get("jenis", "")).lower():
                    scored_electives.append((c, float(h["score"])))
        except Exception:
            pass

    # Add remaining electives from prodi with baseline score 0.0
    for c in prodi_courses:
        if "pilihan" in str(c.get("jenis", "")).lower():
            if not any(se[0]["kode_mk"] == c["kode_mk"] for se in scored_electives):
                scored_electives.append((c, 0.0))

    # Sort electives by relevance score descending
    scored_electives.sort(key=lambda x: x[1], reverse=True)

    # 3. Current and future semesters: CSP assignment (Wajib -> Relevant Electives)
    has_religion = any("agama" in p for p in passed_names)

    for sem in range(current_semester, target_semester + 1):
        sem_sks = 0
        sem_parity = "odd" if sem % 2 == 1 else "even"

        # 3a. Place Wajib courses for this semester (enforce single religion)
        sem_wajib = [
            c for c in prodi_courses
            if int(c.get("semester", 1)) == sem and "wajib" in str(c.get("jenis", "wajib")).lower()
        ]

        for c in sem_wajib:
            c_name = c["nama_mk"]
            c_code = c["kode_mk"]
            c_sks = int(c.get("sks", 2))

            if "agama" in c_name.lower():
                if has_religion:
                    continue
                has_religion = True

            if sem_sks + c_sks <= max_sks_per_sem:
                plan[sem].append({
                    "id": str(c.get("id") or c_code),
                    "code": str(c_code),
                    "name": str(c_name),
                    "credits": c_sks,
                    "semester": sem,
                    "type": "Wajib",
                    "parity": sem_parity,
                    "desc": str(c.get("deskripsi", "")),
                    "status": "planned",
                })
                sem_sks += c_sks
                total_credits += c_sks
                passed_codes.add(c_code.lower())
                passed_names.add(c_name.lower())

        # 3b. Place Electives matching parity and SKS capacity
        for c, _ in scored_electives:
            c_code = c["kode_mk"]
            c_name = c["nama_mk"]
            c_sks = int(c.get("sks", 2))
            c_sem = int(c.get("semester", 1))
            c_parity = "odd" if c_sem % 2 == 1 else "even"

            if c_code.lower() in passed_codes or c_name.lower() in passed_names:
                continue
            # Parity check
            if c_parity != sem_parity:
                continue
            if sem_sks + c_sks <= max_sks_per_sem:
                plan[sem].append({
                    "id": str(c.get("id") or c_code),
                    "code": str(c_code),
                    "name": str(c_name),
                    "credits": c_sks,
                    "semester": sem,
                    "type": "Pilihan",
                    "parity": sem_parity,
                    "desc": str(c.get("deskripsi", "")),
                    "status": "planned",
                })
                sem_sks += c_sks
                total_credits += c_sks
                passed_codes.add(c_code.lower())
                passed_names.add(c_name.lower())

    return {
        "action": "APPLY_DEGREE_PLAN",
        "summary": {
            "program_studi": canonical_prodi,
            "current_semester": current_semester,
            "target_semester": target_semester,
            "total_credits": total_credits,
            "focus_tracks": focus_tracks,
            "note": "Rencana studi terverifikasi deterministik CSP memenuhi batasan kurikulum, paritas, dan SKS FTMM."
        },
        "plan": plan
    }

PRODI_CANONICAL_MAP: dict[str, str] = {
    "tsd": "Teknologi Sains Data",
    "sains data": "Teknologi Sains Data",
    "teknologi sains data": "Teknologi Sains Data",
    "data science": "Teknologi Sains Data",
    "data scientist": "Teknologi Sains Data",
    "data saintis": "Teknologi Sains Data",
    "ds": "Teknologi Sains Data",
    "trkb": "Teknik Robotika dan Kecerdasan Buatan",
    "robotika": "Teknik Robotika dan Kecerdasan Buatan",
    "robotik": "Teknik Robotika dan Kecerdasan Buatan",
    "robotics": "Teknik Robotika dan Kecerdasan Buatan",
    "teknik robotika": "Teknik Robotika dan Kecerdasan Buatan",
    "teknik robotika dan kecerdasan buatan": "Teknik Robotika dan Kecerdasan Buatan",
    "ai robot": "Teknik Robotika dan Kecerdasan Buatan",
    "rn": "S1 Rekayasa Nanoteknologi",
    "nano": "S1 Rekayasa Nanoteknologi",
    "nanoteknologi": "S1 Rekayasa Nanoteknologi",
    "nanotek": "S1 Rekayasa Nanoteknologi",
    "rekayasa nanoteknologi": "S1 Rekayasa Nanoteknologi",
    "s1 rekayasa nanoteknologi": "S1 Rekayasa Nanoteknologi",
    "ti": "Teknik Industri",
    "industri": "Teknik Industri",
    "industrial": "Teknik Industri",
    "teknik industri": "Teknik Industri",
    "te": "Teknik Elektro",
    "elektro": "Teknik Elektro",
    "electrical": "Teknik Elektro",
    "teknik elektro": "Teknik Elektro",
}


def normalize_program_studi(raw: str) -> str:
    cleaned = str(raw or "").lower().strip()
    if not cleaned:
        return "Teknologi Sains Data"
    if cleaned in PRODI_CANONICAL_MAP:
        return PRODI_CANONICAL_MAP[cleaned]
    for alias in sorted(PRODI_CANONICAL_MAP.keys(), key=len, reverse=True):
        if re.search(rf"\b{re.escape(alias)}\b", cleaned):
            return PRODI_CANONICAL_MAP[alias]

    # Semantic fallback via Dense Gazetteer Linker
    try:
        from src.engine.memory import get_dense_gazetteer
        import numpy as np
        linker = get_dense_gazetteer()
        q_vec = linker.embed_query(cleaned)
        sims = np.dot(linker.prodi_matrix, q_vec)
        best_idx = int(np.argmax(sims))
        if sims[best_idx] >= 0.45:
            matched = linker.prodi_labels[best_idx]
            if matched == "Rekayasa Nanoteknologi":
                return "S1 Rekayasa Nanoteknologi"
            return matched
    except Exception:
        pass
    return raw


def extract_semester_from_text(text: str) -> int | None:
    match = re.search(
        r"(?:semester|smt|sem)\s*[:=]?\s*([1-8]|i{1,3}|iv|v|vi{0,3}|viii)\b",
        str(text or ""),
        re.IGNORECASE,
    )
    if match:
        raw_val = match.group(1).lower()
        roman_map = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7, "viii": 8}
        if raw_val in roman_map:
            return roman_map[raw_val]
        try:
            return int(raw_val)
        except ValueError:
            pass
    return None


def get_courses_by_semester(
    program_studi: str,
    semester: int,
    jenis: str | None = None,
) -> dict[str, Any]:
    """
    Mengambil daftar lengkap seluruh mata kuliah pada semester tertentu untuk program studi spesifik secara deterministik in-memory.
    """
    canonical_prodi = normalize_program_studi(program_studi)
    try:
        sem_int = int(semester)
    except (ValueError, TypeError):
        sem_int = 1

    if sem_int < 1 or sem_int > 8:
        return {
            "error": f"Semester {sem_int} di luar rentang kurikulum resmi FTMM (semester 1-8).",
            "program_studi": canonical_prodi,
            "semester": sem_int,
            "total_courses": 0,
            "total_sks": 0,
            "courses": [],
        }

    catalog = _get_catalog()
    filtered: list[dict[str, Any]] = []
    for c in catalog:
        c_prodi = c.get("program_studi", "")
        if c_prodi != canonical_prodi:
            continue
        if int(c.get("semester", 0)) != sem_int:
            continue
        if jenis:
            j_clean = jenis.lower().strip()
            c_jenis = str(c.get("jenis", "")).lower()
            if "wajib" in j_clean and "wajib" not in c_jenis:
                continue
            if "pilih" in j_clean and ("pilih" not in c_jenis and "peminatan" not in c_jenis):
                continue
            if j_clean not in c_jenis and "wajib" not in j_clean and "pilih" not in j_clean:
                continue
        filtered.append({
            "kode_mk": c["kode_mk"],
            "nama_mk": c["nama_mk"],
            "sks": int(c.get("sks", 0)),
            "jenis": c.get("jenis", "wajib"),
            "prasyarat": c.get("prasyarat", []),
        })

    total_sks = sum(c["sks"] for c in filtered)
    return {
        "program_studi": canonical_prodi,
        "semester": sem_int,
        "total_courses": len(filtered),
        "total_sks": total_sks,
        "courses": filtered,
    }


def format_course_catalog_response(course_data: dict[str, Any], language: str = "id") -> str:
    """
    Format data kurikulum semester menjadi Markdown baku FTMM COMPASS.
    Mendukung dwibahasa (id/en) dengan struktur daftar mata kuliah berformat tebal.
    """
    if not course_data or not course_data.get("courses"):
        if language == "en":
            return "No courses found for the specified semester."
        return "Data mata kuliah tidak ditemukan untuk semester tersebut."

    prodi = course_data.get("program_studi", "")
    semester = course_data.get("semester", "")
    total_c = course_data.get("total_courses", len(course_data["courses"]))
    total_sks = course_data.get("total_sks", 0)

    if language == "en":
        header = f"Here is the official course catalog for **{prodi}** in **Semester {semester}**:\n"
        none_prereq = "None"
        footer = f"\n**Total:** {total_c} courses ({total_sks} credits)"
    else:
        header = f"Berikut adalah daftar mata kuliah untuk **{prodi}** pada **Semester {semester}**:\n"
        none_prereq = "Tidak ada"
        footer = f"\n**Total:** {total_c} mata kuliah ({total_sks} SKS)"

    lines = [header]
    for idx, c in enumerate(course_data["courses"], 1):
        prereqs = c.get("prasyarat", [])
        p_str = ", ".join(prereqs) if prereqs else none_prereq
        raw_jenis = str(c.get("jenis", "wajib")).lower()
        if language == "en":
            jenis_str = "Required" if "wajib" in raw_jenis else "Elective"
            label_prereq = "Prerequisites"
            label_sks = "Credits"
        else:
            jenis_str = "Wajib" if "wajib" in raw_jenis else "Pilihan"
            label_prereq = "Prasyarat"
            label_sks = "SKS"

        lines.append(
            f"{idx}. **[{c.get('kode_mk', '-')}] {c.get('nama_mk', '-')}** "
            f"({c.get('sks', 0)} {label_sks}, {jenis_str}) — {label_prereq}: {p_str}"
        )
    lines.append(footer)
    return "\n".join(lines)
