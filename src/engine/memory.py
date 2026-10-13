import os
import pickle
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import numpy as np
import ollama

DEFAULT_TTL_SECONDS = 7200  # 2 hours
DEFAULT_MAX_WINDOW_TURNS = 6  # 6 turns = 12 messages
DEFAULT_EMBED_MODEL = "nomic-embed-text"
GAZETTEER_CACHE_PATH = Path("data/dense_gazetteer.npz")

COURSE_CODE_PATTERN = re.compile(r"\b([A-Z]{3}\s?[0-9]{3})\b", re.IGNORECASE)
SEMESTER_PATTERN = re.compile(
    r"\b(?:semester|sem|smt|semseter)\s*([1-8]|i{1,3}|iv|v|vi{0,3}|viii)\b",
    re.IGNORECASE,
)

ROMAN_MAP = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7, "viii": 8}


class DenseGazetteerLinker:
    """In-memory vector gazetteer matcher using dot product over normalized embeddings."""

    def __init__(self, embed_model: str = DEFAULT_EMBED_MODEL, cache_path: Path | str = GAZETTEER_CACHE_PATH):
        self.embed_model = embed_model
        self.cache_path = Path(cache_path)
        self.prodi_matrix: np.ndarray | None = None
        self.prodi_labels: list[str] = []
        self.focus_matrix: np.ndarray | None = None
        self.focus_labels: list[str] = []
        self.ellipsis_matrix: np.ndarray | None = None
        self.courses_matrix: np.ndarray | None = None
        self.course_names: list[str] = []
        self._initialized = False

    def initialize(self):
        if self._initialized:
            return
        if not self.cache_path.exists():
            raise FileNotFoundError(f"Gazetteer cache not found at {self.cache_path}")
        data = np.load(self.cache_path)
        self.prodi_matrix = np.array(data["prodi_matrix"], dtype=np.float32)
        self.prodi_labels = [str(x) for x in data["prodi_labels"]]
        self.focus_matrix = np.array(data["focus_matrix"], dtype=np.float32)
        self.focus_labels = [str(x) for x in data["focus_labels"]]
        self.ellipsis_matrix = np.array(data["ellipsis_matrix"], dtype=np.float32)
        self.prodi_intent_matrix = (
            np.array(data["prodi_intent_matrix"], dtype=np.float32)
            if "prodi_intent_matrix" in data
            else None
        )
        self.courses_matrix = np.array(data["courses_matrix"], dtype=np.float32)
        self.course_names = [str(x) for x in data["course_names"]]
        self._initialized = True

    def embed_query(self, query: str) -> np.ndarray:
        resp = ollama.embed(model=self.embed_model, input=query)
        q_vec = np.array(resp["embeddings"][0], dtype=np.float32)
        norm = np.linalg.norm(q_vec)
        if norm > 0:
            q_vec = q_vec / norm
        return q_vec

    def match_prodi(self, q_vec: np.ndarray, threshold: float = 0.55) -> str | None:
        if not self._initialized:
            self.initialize()
        if self.prodi_intent_matrix is not None:
            sim_intent = np.dot(self.prodi_intent_matrix, q_vec)
            if sim_intent[0] < sim_intent[1] + 0.02:
                return None
        sims = np.dot(self.prodi_matrix, q_vec)
        best_idx = int(np.argmax(sims))
        if sims[best_idx] >= threshold:
            return self.prodi_labels[best_idx]
        return None

    def match_focus(self, q_vec: np.ndarray, threshold: float = 0.60) -> str | None:
        if not self._initialized:
            self.initialize()
        sims = np.dot(self.focus_matrix, q_vec)
        best_idx = int(np.argmax(sims))
        if sims[best_idx] >= threshold:
            return self.focus_labels[best_idx]
        return None

    def is_ellipsis_or_reference(self, q_vec: np.ndarray) -> bool:
        if not self._initialized:
            self.initialize()
        sims = np.dot(self.ellipsis_matrix, q_vec)
        return bool(sims[0] > sims[1] or sims[0] >= 0.70)

    def match_course(self, q_vec: np.ndarray, threshold: float = 0.72) -> str | None:
        if not self._initialized:
            self.initialize()
        sims = np.dot(self.courses_matrix, q_vec)
        best_idx = int(np.argmax(sims))
        if sims[best_idx] >= threshold:
            return self.course_names[best_idx]
        return None

    def find_all_courses_in_text(self, text: str) -> list[str]:
        if not self._initialized:
            self.initialize()
        t_lower = text.lower()
        matched = []
        for name in self.course_names:
            if len(name) >= 4 and name.lower() in t_lower:
                matched.append(name)
        return matched


_gazetteer_instance: DenseGazetteerLinker | None = None


def get_dense_gazetteer() -> DenseGazetteerLinker:
    global _gazetteer_instance
    if _gazetteer_instance is None:
        _gazetteer_instance = DenseGazetteerLinker()
        _gazetteer_instance.initialize()
    return _gazetteer_instance
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
        gazetteer: DenseGazetteerLinker | None = None,
    ):
        self.max_window_turns = max_window_turns
        self.ttl_seconds = ttl_seconds
        self.sessions: dict[str, SessionState] = {}
        self.gazetteer = gazetteer or get_dense_gazetteer()

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

    def extract_profile_updates(
        self, text: str, query_vec: np.ndarray | None = None
    ) -> dict[str, Any]:
        updates: dict[str, Any] = {}
        q_vec = query_vec if query_vec is not None else self.gazetteer.embed_query(text)

        prodi = self.gazetteer.match_prodi(q_vec)
        if prodi:
            updates["program_studi"] = prodi

        m_sem = SEMESTER_PATTERN.search(text)
        if m_sem:
            val_raw = m_sem.group(1).lower()
            updates["semester_saat_ini"] = ROMAN_MAP.get(val_raw, None) or (
                int(val_raw) if val_raw.isdigit() else None
            )

        focus = self.gazetteer.match_focus(q_vec)
        if focus:
            updates.setdefault("minat_fokus", []).append(focus)

        passed = self.gazetteer.find_all_courses_in_text(text)
        if passed:
            updates["riwayat_matkul_lulus"] = passed

        return updates

    def extract_course_entity(
        self, text: str, query_vec: np.ndarray | None = None
    ) -> str | None:
        m_code = COURSE_CODE_PATTERN.search(text)
        if m_code:
            return m_code.group(1).upper()

        courses = self.gazetteer.find_all_courses_in_text(text)
        if courses:
            return courses[0]

        q_vec = query_vec if query_vec is not None else self.gazetteer.embed_query(text)
        return self.gazetteer.match_course(q_vec)

    def resolve_anaphora(
        self, session_id: str, query: str, query_vec: np.ndarray | None = None
    ) -> str:
        session = self.get_or_create_session(session_id)
        last_entity = session.last_discussed_course
        if not last_entity:
            return query

        q_vec = query_vec if query_vec is not None else self.gazetteer.embed_query(query)
        is_ref = self.gazetteer.is_ellipsis_or_reference(q_vec)
        if is_ref and last_entity.lower() not in query.lower():
            q_clean = query
            for word in ["mata kuliah tersebut", "matkul tersebut", "mata kuliah itu", "matkul itu", "tersebut", "itu"]:
                if word in q_clean.lower():
                    idx = q_clean.lower().find(word)
                    q_clean = q_clean[:idx] + f"mata kuliah {last_entity}" + q_clean[idx + len(word):]
                    return q_clean
            return f"{query} (mata kuliah {last_entity})"
        return query

    def contextualize_query(
        self,
        session_id: str,
        query: str,
        query_vec: np.ndarray | None = None,
        decider_category: str | None = None,
    ) -> str:
        session = self.get_or_create_session(session_id)
        q_vec = query_vec if query_vec is not None else self.gazetteer.embed_query(query)

        is_ref = self.gazetteer.is_ellipsis_or_reference(q_vec)

        active_prodi = session.student_profile.get("program_studi")
        active_sem = (
            session.last_discussed_semester
            or session.student_profile.get("semester_saat_ini")
        )
        active_course = session.last_discussed_course

        if is_ref and (active_prodi or active_sem or active_course):
            context_parts = []
            if active_prodi:
                context_parts.append(f"Prodi={active_prodi}")
            if active_sem:
                context_parts.append(f"Semester={active_sem}")
            if active_course:
                context_parts.append(f"Matkul={active_course}")
            context_tag = f"[KONTEKS AKTIF: {', '.join(context_parts)}]"
            resolved = f"{query} {context_tag}"
            return resolved

        return self.resolve_anaphora(session_id, query, q_vec)

    def add_turn(
        self,
        session_id: str,
        user_message: str,
        assistant_response: str,
        query_vec: np.ndarray | None = None,
        decider_category: str | None = None,
    ) -> SessionState:
        session = self.get_or_create_session(session_id)
        q_vec = query_vec if query_vec is not None else self.gazetteer.embed_query(user_message)

        prof_updates = self.extract_profile_updates(user_message, query_vec=q_vec)
        if "program_studi" in prof_updates:
            session.student_profile["program_studi"] = prof_updates["program_studi"]
        if "semester_saat_ini" in prof_updates:
            session.student_profile["semester_saat_ini"] = prof_updates["semester_saat_ini"]
            session.last_discussed_semester = prof_updates["semester_saat_ini"]
        if "minat_fokus" in prof_updates:
            curr_focus = set(session.student_profile.get("minat_fokus", []))
            curr_focus.update(prof_updates["minat_fokus"])
            session.student_profile["minat_fokus"] = sorted(list(curr_focus))
        if "riwayat_matkul_lulus" in prof_updates:
            curr_passed = set(session.student_profile.get("riwayat_matkul_lulus", []))
            curr_passed.update(prof_updates["riwayat_matkul_lulus"])
            session.student_profile["riwayat_matkul_lulus"] = sorted(list(curr_passed))

        if decider_category:
            session.last_intent_topic = decider_category

        entity = self.extract_course_entity(user_message, query_vec=q_vec)
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
