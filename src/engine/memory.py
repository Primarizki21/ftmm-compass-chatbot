import re
import time
from dataclasses import dataclass, field
from typing import Any

DEFAULT_TTL_SECONDS = 7200  # 2 hours
DEFAULT_MAX_WINDOW_TURNS = 6  # 6 turns = 12 messages

PRODI_PATTERNS = [
    (r"\b(teknologi\s+sains\s+data|sains\s+data|data\s+science|tsd)\b", "Teknologi Sains Data"),
    (r"\b(teknik\s+robotika(\s+dan\s+kecerdasan\s+buatan)?|robotika|robotik|trkb|robotics)\b", "Teknik Robotika dan Kecerdasan Buatan"),
    (r"\b(rekayasa\s+nanoteknologi|nanoteknologi|nanotek|rn|nano)\b", "Rekayasa Nanoteknologi"),
    (r"\b(teknik\s+industri|industri|industrial(\s+engineering)?|ti)\b", "Teknik Industri"),
    (r"\b(teknik\s+elektro|elektro|electrical(\s+engineering)?|te)\b", "Teknik Elektro"),
]

SEMESTER_PATTERN = re.compile(
    r"\b(?:semester|sem|smt|semseter)\s*([1-8]|i{1,3}|iv|v|vi{0,3}|viii)\b",
    re.IGNORECASE,
)

FOCUS_PATTERNS = [
    (r"\b(rantai\s+pasok|logistik|supply\s+chain)\b", "Rantai Pasok"),
    (r"\b(manufaktur|sistem\s+manufaktur)\b", "Sistem Manufaktur"),
    (r"\b(big\s+data|data\s+analytics|analitika\s+data)\b", "Big Data"),
    (r"\b(kecerdasan\s+buatan|artificial\s+intelligence|ai)\b", "Kecerdasan Buatan"),
    (r"\b(nanomedisin|nanobiologi|biomedis)\b", "Nanomedisin"),
    (r"\b(sistem\s+tenaga|power\s+systems)\b", "Sistem Tenaga"),
]

DEMONSTRATIVE_PATTERN = re.compile(
    r"\b(mata\s+kuliah\s+tersebut|matkul\s+tersebut|mata\s+kuliah\s+itu|matkul\s+itu|\bitu\b|\btersebut\b)\b",
    re.IGNORECASE,
)
RELATIONAL_NYA_PATTERN = re.compile(
    r"\b(dosen(\s*pengampu)?|prasyarat|beban\s+sks|sks|silabus|jadwal)nya\b",
    re.IGNORECASE,
)


@dataclass
class SessionState:
    session_id: str
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    messages: list[dict[str, str]] = field(default_factory=list)
    student_profile: dict[str, Any] = field(
        default_factory=lambda: {
            "program_studi": None,
            "semester_saat_ini": None,
            "riwayat_matkul_lulus": [],
            "minat_fokus": [],
            "target_kelulusan_semester": 8,
        }
    )
    last_discussed_course: str | None = None
    last_discussed_semester: int | None = None
    last_intent_topic: str | None = None
    cumulative_risk: float = 0.0

class ConversationMemoryManager:
    """Manages multi-turn conversation history, sliding window buffers,

    student profile accumulation, and anaphora resolution per session_id.
    """

    def __init__(
        self,
        max_window_turns: int = DEFAULT_MAX_WINDOW_TURNS,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
    ):
        self.max_window_turns = max_window_turns
        self.ttl_seconds = ttl_seconds
        self.sessions: dict[str, SessionState] = {}

    def get_or_create_session(self, session_id: str) -> SessionState:
        now = time.time()
        session = self.sessions.get(session_id)
        if session is not None:
            if now - session.last_active > self.ttl_seconds:
                self.sessions.pop(session_id, None)
                session = None
            else:
                session.last_active = now
                return session

        new_session = SessionState(session_id=session_id, created_at=now, last_active=now)
        self.sessions[session_id] = new_session
        return new_session

    def extract_profile_updates(self, text: str) -> dict[str, Any]:
        updates: dict[str, Any] = {}
        t_clean = text.lower()

        for pattern, prodi_name in PRODI_PATTERNS:
            if re.search(pattern, t_clean):
                updates["program_studi"] = prodi_name
                break

        m_sem = SEMESTER_PATTERN.search(t_clean)
        if m_sem:
            val_raw = m_sem.group(1).lower()
            roman_map = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7, "viii": 8}
            updates["semester_saat_ini"] = roman_map.get(val_raw, None) or (int(val_raw) if val_raw.isdigit() else None)
        for pattern, focus_name in FOCUS_PATTERNS:
            if re.search(pattern, t_clean):
                updates.setdefault("minat_fokus", []).append(focus_name)

        # Detect completed/passed courses
        m_passed = re.search(r"\b(?:sudah\s+lulus|telah\s+lulus|lulus)\s+([^.?!,]+)", t_clean)
        if m_passed:
            raw_courses = m_passed.group(1).split(" dan ")
            cleaned_courses = []
            for c in raw_courses:
                for sub in c.split(","):
                    sc = sub.strip().title()
                    if len(sc) > 3 and sc not in cleaned_courses:
                        cleaned_courses.append(sc)
            if cleaned_courses:
                updates["riwayat_matkul_lulus"] = cleaned_courses

        return updates

    def extract_course_entity(self, text: str) -> str | None:
        known_courses = [
            "Data Mining I",
            "Data Mining",
            "Pembelajaran Mesin",
            "Sintesis Material Anorganik",
            "Dasar Sistem Tenaga",
            "Alpro I",
            "Alpro II",
            "Struktur Data",
            "Jaringan Komputer",
            "Simulasi Sistem Industri",
            "Rangkaian Listrik",
            "Elektronika Dasar",
            "Kalkulus I",
            "Kalkulus II",
        ]
        for c in known_courses:
            if re.search(rf"\b{re.escape(c)}\b", text, re.IGNORECASE):
                return c

        m_code = re.search(r"\b([A-Z]{3}\s?[0-9]{3})\b", text)
        if m_code:
            return m_code.group(1).upper()

        m_matkul = re.search(
            r"\b(?:mata\s+kuliah|matkul)\s+([A-Za-z0-9\s]+?)(?:\s+(?:semester|prasyarat|ini|itu|ga|bro|ya|\?|\.|$))",
            text,
            re.IGNORECASE,
        )
        if m_matkul:
            cand = m_matkul.group(1).strip().title()
            if len(cand) > 2 and cand.lower() not in {"itu", "tersebut", "ini", "apa", "berapa"}:
                return cand

        return None

    def resolve_anaphora(self, session_id: str, query: str) -> str:
        session = self.get_or_create_session(session_id)
        last_entity = session.last_discussed_course

        if not last_entity:
            return query

        resolved = query
        if DEMONSTRATIVE_PATTERN.search(resolved):
            resolved = DEMONSTRATIVE_PATTERN.sub(f"mata kuliah {last_entity}", resolved)
        elif RELATIONAL_NYA_PATTERN.search(resolved):
            resolved = RELATIONAL_NYA_PATTERN.sub(r"\1", resolved)
            resolved = f"{resolved} mata kuliah {last_entity}"
        elif re.search(r"\b(dosen|prasyarat|sks)\b", resolved, re.IGNORECASE) and last_entity.lower() not in resolved.lower():
            resolved = f"{resolved} (mata kuliah {last_entity})"

        return resolved

    def contextualize_query(self, session_id: str, query: str) -> str:
        session = self.get_or_create_session(session_id)
        resolved = self.resolve_anaphora(session_id, query)

        active_sem = session.last_discussed_semester or session.student_profile.get("semester_saat_ini")
        
        prodi_mentioned = re.search(
            r"\b(teknologi\s+sains\s+data|sains\s+data|data\s+science|tsd|ds|"
            r"teknik\s+robotika(\s+dan\s+kecerdasan\s+buatan)?|robotika|trkb|robotics|"
            r"rekayasa\s+nanoteknologi|nanoteknologi|rn|nano|"
            r"teknik\s+industri|industrial|ti|"
            r"teknik\s+elektro|electrical|te)\b",
            resolved,
            re.IGNORECASE,
        )
        has_semester = SEMESTER_PATTERN.search(resolved)

        if prodi_mentioned and not has_semester and active_sem is not None:
            if session.last_intent_topic == "SEMESTER_COURSES":
                if not re.search(r"\b(mata\s*kuliah|matkul|courses?)\b", resolved, re.IGNORECASE):
                    resolved = f"{resolved} (mata kuliah semester {active_sem})"
                else:
                    resolved = f"{resolved} semester {active_sem}"

        return resolved

    def add_turn(
        self,
        session_id: str,
        user_message: str,
        assistant_response: str,
    ) -> SessionState:
        session = self.get_or_create_session(session_id)

        # Update profile
        prof_updates = self.extract_profile_updates(user_message)
        if "program_studi" in prof_updates:
            session.student_profile["program_studi"] = prof_updates["program_studi"]
        if "semester_saat_ini" in prof_updates:
            session.student_profile["semester_saat_ini"] = prof_updates["semester_saat_ini"]
        if "minat_fokus" in prof_updates:
            curr_focus = set(session.student_profile.get("minat_fokus", []))
            curr_focus.update(prof_updates["minat_fokus"])
            session.student_profile["minat_fokus"] = sorted(list(curr_focus))
        if "riwayat_matkul_lulus" in prof_updates:
            curr_passed = set(session.student_profile.get("riwayat_matkul_lulus", []))
            curr_passed.update(prof_updates["riwayat_matkul_lulus"])
            session.student_profile["riwayat_matkul_lulus"] = sorted(list(curr_passed))


        # Update semester & topic tracking
        if "semester_saat_ini" in prof_updates:
            session.last_discussed_semester = prof_updates["semester_saat_ini"]

        if re.search(r"(?:mata\s*kuliah|matkul|courses?).*?(?:semester|sem|smt)\s*[1-8]|(?:semester|sem|smt)\s*[1-8].*?(?:mata\s*kuliah|matkul)", user_message, re.IGNORECASE):
            session.last_intent_topic = "SEMESTER_COURSES"
        elif re.search(r"\b(prasyarat|syarat)\b", user_message, re.IGNORECASE):
            session.last_intent_topic = "PREREQUISITE"
        elif re.search(r"\b(batas\s+sks|jatah\s+sks|beban\s+sks|maksimal\s+sks)\b", user_message, re.IGNORECASE):
            session.last_intent_topic = "SKS_LIMIT"
        elif re.search(r"\b(rencana\s+studi|degree\s+plan|plotting\s+krs)\b", user_message, re.IGNORECASE):
            session.last_intent_topic = "DEGREE_PLAN"
        # Update entity tracking
        entity = self.extract_course_entity(user_message)
        if entity:
            session.last_discussed_course = entity

        session.messages.append({"role": "user", "content": user_message})
        session.messages.append({"role": "assistant", "content": assistant_response})
        session.last_active = time.time()

        return session

    def get_context_for_llm(
        self,
        session_id: str,
    ) -> tuple[list[dict[str, str]], dict[str, Any]]:
        session = self.get_or_create_session(session_id)
        max_messages = self.max_window_turns * 2
        sliding_messages = session.messages[-max_messages:] if len(session.messages) > max_messages else list(session.messages)
        return sliding_messages, dict(session.student_profile)

    def clear_session(self, session_id: str):
        self.sessions.pop(session_id, None)

    def cleanup_expired_sessions(self) -> int:
        now = time.time()
        expired = [sid for sid, s in self.sessions.items() if now - s.last_active > self.ttl_seconds]
        for sid in expired:
            del self.sessions[sid]
        return len(expired)
