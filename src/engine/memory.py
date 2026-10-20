import json
import os
import pickle
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import numpy as np
import ollama
from src.engine.base import get_ollama_client

DEFAULT_TTL_SECONDS = 7200  # 2 hours
DEFAULT_MAX_WINDOW_TURNS = 6  # 6 turns = 12 messages
DEFAULT_EMBED_MODEL = os.getenv("EMBED_MODEL_NAME", "nomic-embed-text-v2-moe")
GAZETTEER_CACHE_PATH = Path("data/dense_gazetteer.npz")

COURSE_CODE_PATTERN = re.compile(r"\b([A-Z]{3}\s?[0-9]{3})\b", re.IGNORECASE)
SEMESTER_PATTERN = re.compile(
    r"\b(?:semester|sem|smt|semseter)[\s=:]*([1-8]|i{1,3}|iv|v|vi{0,3}|viii)\b",
    re.IGNORECASE,
)

ROMAN_MAP = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7, "viii": 8}


def extract_semester_from_text(text: str) -> int | None:
    """Mengekstrak nilai semester numerik (1-8) atau Romawi (I-VIII) dari teks pengguna."""
    match = SEMESTER_PATTERN.search(str(text or ""))
    if match:
        raw_val = match.group(1).lower()
        if raw_val in ROMAN_MAP:
            return ROMAN_MAP[raw_val]
        if raw_val.isdigit():
            val_int = int(raw_val)
            if 1 <= val_int <= 8:
                return val_int
    return None

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
        try:
            resp = get_ollama_client().embed(model=self.embed_model, input=query, keep_alive=-1)
            q_vec = np.array(resp["embeddings"][0], dtype=np.float32)
            norm = np.linalg.norm(q_vec)
            if norm > 0:
                q_vec = q_vec / norm
            return q_vec
        except Exception:
            dim = self.courses_matrix.shape[1] if (self._initialized and getattr(self, "courses_matrix", None) is not None) else 768
            return np.zeros(dim, dtype=np.float32)

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


class BaseSessionStore:
    def get(self, session_id: str) -> SessionState | None:
        raise NotImplementedError

    def save(self, session: SessionState) -> None:
        raise NotImplementedError

    def delete(self, session_id: str) -> None:
        raise NotImplementedError

    def cleanup_expired(self, ttl_seconds: int) -> int:
        raise NotImplementedError

    def all_sessions(self) -> dict[str, SessionState]:
        raise NotImplementedError


class InMemorySessionStore(BaseSessionStore):
    def __init__(self) -> None:
        self.sessions: dict[str, SessionState] = {}

    def get(self, session_id: str) -> SessionState | None:
        return self.sessions.get(session_id)

    def save(self, session: SessionState) -> None:
        self.sessions[session.session_id] = session

    def delete(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)

    def cleanup_expired(self, ttl_seconds: int) -> int:
        now = time.time()
        expired = [sid for sid, s in self.sessions.items() if now - s.last_active > ttl_seconds]
        for sid in expired:
            del self.sessions[sid]
        return len(expired)

    def all_sessions(self) -> dict[str, SessionState]:
        return self.sessions


class SQLiteSessionStore(BaseSessionStore):
    def __init__(self, db_path: str | Path | None = None) -> None:
        effective_path = db_path or os.getenv("SESSION_DB_PATH", "data/sessions/sessions.db")
        self.db_path = Path(effective_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def _init_db(self) -> None:
        with self._get_conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at REAL,
                    last_active REAL,
                    payload TEXT
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_active ON sessions(last_active);")

    def get(self, session_id: str) -> SessionState | None:
        with self._get_conn() as conn:
            cursor = conn.execute("SELECT payload FROM sessions WHERE session_id = ?", (session_id,))
            row = cursor.fetchone()
            if not row or not row[0]:
                return None
            try:
                data = json.loads(row[0])
                return SessionState(
                    session_id=data["session_id"],
                    created_at=data.get("created_at", time.time()),
                    last_active=data.get("last_active", time.time()),
                    messages=data.get("messages", []),
                    student_profile=data.get("student_profile", {}),
                    last_discussed_course=data.get("last_discussed_course"),
                    last_discussed_semester=data.get("last_discussed_semester"),
                    last_intent_topic=data.get("last_intent_topic"),
                    cumulative_risk=data.get("cumulative_risk", 0.0),
                )
            except Exception:
                return None

    def save(self, session: SessionState) -> None:
        payload = json.dumps(
            {
                "session_id": session.session_id,
                "created_at": session.created_at,
                "last_active": session.last_active,
                "messages": session.messages,
                "student_profile": session.student_profile,
                "last_discussed_course": session.last_discussed_course,
                "last_discussed_semester": session.last_discussed_semester,
                "last_intent_topic": session.last_intent_topic,
                "cumulative_risk": session.cumulative_risk,
            },
            ensure_ascii=False,
        )
        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT INTO sessions (session_id, created_at, last_active, payload)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    last_active = excluded.last_active,
                    payload = excluded.payload
                """,
                (session.session_id, session.created_at, session.last_active, payload),
            )

    def delete(self, session_id: str) -> None:
        with self._get_conn() as conn:
            conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))

    def cleanup_expired(self, ttl_seconds: int) -> int:
        cutoff = time.time() - ttl_seconds
        with self._get_conn() as conn:
            cursor = conn.execute("DELETE FROM sessions WHERE last_active < ?", (cutoff,))
            return cursor.rowcount

    def all_sessions(self) -> dict[str, SessionState]:
        res: dict[str, SessionState] = {}
        with self._get_conn() as conn:
            cursor = conn.execute("SELECT session_id, payload FROM sessions")
            for row in cursor.fetchall():
                try:
                    data = json.loads(row[1])
                    res[row[0]] = SessionState(
                        session_id=data["session_id"],
                        created_at=data.get("created_at", time.time()),
                        last_active=data.get("last_active", time.time()),
                        messages=data.get("messages", []),
                        student_profile=data.get("student_profile", {}),
                        last_discussed_course=data.get("last_discussed_course"),
                        last_discussed_semester=data.get("last_discussed_semester"),
                        last_intent_topic=data.get("last_intent_topic"),
                        cumulative_risk=data.get("cumulative_risk", 0.0),
                    )
                except Exception:
                    continue
        return res

class ConversationMemoryManager:
    """Manages multi-turn conversation history, sliding window buffers,

    student profile accumulation, and anaphora resolution per session_id.
    """

    def __init__(
        self,
        max_window_turns: int = DEFAULT_MAX_WINDOW_TURNS,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        gazetteer: DenseGazetteerLinker | None = None,
        store: BaseSessionStore | None = None,
    ):
        self.max_window_turns = max_window_turns
        self.ttl_seconds = ttl_seconds
        self.gazetteer = gazetteer or get_dense_gazetteer()
        if store is not None:
            self.store = store
        else:
            backend = os.getenv("SESSION_STORE_BACKEND", "memory").strip().lower()
            if backend == "sqlite":
                try:
                    self.store = SQLiteSessionStore()
                except Exception:
                    self.store = InMemorySessionStore()
            else:
                self.store = InMemorySessionStore()

    @property
    def sessions(self) -> dict[str, SessionState]:
        return self.store.all_sessions()

    def get_or_create_session(self, session_id: str) -> SessionState:
        now = time.time()
        session = self.store.get(session_id)
        if session is not None:
            if now - session.last_active > self.ttl_seconds:
                self.store.delete(session_id)
                session = None
            else:
                session.last_active = now
                self.store.save(session)
                return session

        new_session = SessionState(session_id=session_id, created_at=now, last_active=now)
        self.store.save(new_session)
        return new_session

    def extract_profile_updates(
        self, text: str, query_vec: np.ndarray | None = None
    ) -> dict[str, Any]:
        updates: dict[str, Any] = {}
        q_vec = query_vec if query_vec is not None else self.gazetteer.embed_query(text)

        from src.engine.tools.academic_tools import normalize_program_studi
        norm_p = normalize_program_studi(text)
        if norm_p != text:
            updates["program_studi"] = norm_p
        else:
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

        current_prodi_in_query = self.gazetteer.match_prodi(q_vec)
        current_sem_in_query = SEMESTER_PATTERN.search(query)

        active_prodi = session.student_profile.get("program_studi") if not current_prodi_in_query else None
        active_sem = (
            (session.last_discussed_semester or session.student_profile.get("semester_saat_ini"))
            if not current_sem_in_query
            else None
        )
        active_course = session.last_discussed_course if not current_prodi_in_query else None
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
        self.store.save(session)

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
        self.store.delete(session_id)

    def cleanup_expired_sessions(self) -> int:
        return self.store.cleanup_expired(self.ttl_seconds)
