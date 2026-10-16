import os
import time
from typing import Any
import numpy as np
import ollama

from src.engine.base import EngineResult, DEFAULT_NUM_CTX, get_ollama_client
from src.guardrails.tier1_filter import Tier1DeterministicFilter
from src.guardrails.tier2_semantic_gate import Tier2SemanticGate
from src.engine.decider import System1Decider, DecisionCategory
from src.engine.memory import ConversationMemoryManager
from src.engine.react_agent import ReActAgentEngine
from src.knowledge.cag_store import get_cag_system_prompt
from src.engine.normalizer import get_academic_normalizer

DEFAULT_MODEL = "qwen2.5:3b"

TIER1_REFUSAL_MESSAGE = (
    "Permintaan tidak dapat diproses karena melanggar kebijakan keamanan sistem akademik FTMM UNAIR."
)

TIER2_REFUSAL_MESSAGE = (
    "Maaf, sebagai asisten akademik FTMM UNAIR, permintaan di luar cakupan atau kebijakan keamanan "
    "sistem akademik tidak dapat diproses. Saya hanya dapat membantu pertanyaan terkait perkuliahan, "
    "kurikulum, silabus mata kuliah, regulasi akademik (SKS, PKL, Skripsi, Yudisium), dan perencanaan studi di FTMM."
)
TIER2_HOMEWORK_REFUSAL_MESSAGE = (
    "Maaf, sebagai asisten akademik FTMM UNAIR, saya berfokus pada informasi kurikulum, "
    "silabus mata kuliah, prasyarat, dan perencanaan studi di FTMM. Saya tidak dapat mengerjakan "
    "tugas/soal, menurunkan rumus, atau menuliskan kode program mandiri. Silakan konsultasikan topik "
    "materi terkait dengan silabus mata kuliah resmi FTMM."
)


class UnifiedChatEngine:
    """
    Unified Orchestration Engine untuk Chatbot FTMM COMPASS.
    Mengintegrasikan 4 pilar riset backend:
    1. Multi-Turn Memory & Contextualizer (Sliding window + Anaphora/Ellipsis resolution)
    2. Cascaded Guardrails (Tier 1 Regex Filter + Tier 2 Semantic OOD Gate)
    3. System-1 Decider (Fast tool pruning)
    4. ReAct Agent Engine (Deterministic academic tools: catalog, DAG, SKS, degree planner)
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        memory_manager: ConversationMemoryManager | None = None,
        embed_model: str | None = None,
    ):
        self.model = model
        self.embed_model = embed_model or os.getenv("EMBED_MODEL_NAME", "nomic-embed-text-v2-moe")
        self.memory = memory_manager or ConversationMemoryManager()
        self.tier1_filter = Tier1DeterministicFilter()
        self.tier2_gate = Tier2SemanticGate(embed_model=self.embed_model)
        self.tier2_gate.initialize()
        self.decider = System1Decider(embed_model=self.embed_model)
        self.decider.initialize()
        self.react_engine = ReActAgentEngine(model=self.model)
        self.normalizer = get_academic_normalizer()

    def _embed_once(self, text: str) -> tuple[np.ndarray, float]:
        t_start = time.perf_counter()
        resp_embed = get_ollama_client().embed(model=self.embed_model, input=text, keep_alive=-1)
        q_vec = np.array(resp_embed["embeddings"][0], dtype=np.float32)
        norm = np.linalg.norm(q_vec)
        if norm > 0:
            q_vec = q_vec / norm
        dt_ms = (time.perf_counter() - t_start) * 1000.0
        return q_vec, dt_ms

    def generate_answer(
        self,
        query: str,
        history: list[dict[str, str]] | None = None,
        student_profile: dict[str, Any] | None = None,
        session_id: str = "default_session",
    ) -> EngineResult:
        t0 = time.perf_counter()
        q_raw = query.strip()

        # Step 1: Tier 1 Instant Regex Defense
        t1_res = self.tier1_filter.check(q_raw)
        if t1_res.blocked:
            total_dt = (time.perf_counter() - t0) * 1000.0
            return EngineResult(
                answer=TIER1_REFUSAL_MESSAGE,
                architecture="UnifiedChatEngine (Tier-1 Blocked)",
                retrieved_chunks=[],
                retrieval_latency_ms=0.0,
                generation_latency_ms=0.0,
                total_latency_ms=total_dt,
                tokens_generated=len(TIER1_REFUSAL_MESSAGE.split()),
                routing_mode="GUARDED",
                metadata={"tier": "TIER_1", "category": t1_res.category, "reason": t1_res.reason},
            )

        # Step 1b: Upstream Academic Normalization (slang O(1), STT phonetic, Bounded Levenshtein d<=2)
        norm_res = self.normalizer.normalize(q_raw)
        q_clean = norm_res.normalized_query

        # Step 2: 1x Shared Embedding for Normalized Query Vector
        q_vec, shared_embed_ms = self._embed_once(q_clean)
        # Step 3: Session State Sync & Multi-Turn Contextualization
        session = self.memory.get_or_create_session(session_id)
        if history and not session.messages:
            for m in history:
                session.messages.append(m)
                if m.get("role") == "user":
                    updates = self.memory.extract_profile_updates(m["content"])
                    if "semester_saat_ini" in updates:
                        session.last_discussed_semester = updates["semester_saat_ini"]
                        session.student_profile["semester_saat_ini"] = updates["semester_saat_ini"]
                    if "program_studi" in updates:
                        session.student_profile["program_studi"] = updates["program_studi"]

        if student_profile:
            for k, v in student_profile.items():
                if v is not None and v != [] and not session.student_profile.get(k):
                    session.student_profile[k] = v
            if student_profile.get("semester_saat_ini") and not session.last_discussed_semester:
                session.last_discussed_semester = student_profile["semester_saat_ini"]

        contextualized_q = self.memory.contextualize_query(
            session_id, q_clean, query_vec=q_vec
        )
        if contextualized_q != q_clean:
            q_vec, shared_embed_ms = self._embed_once(contextualized_q)
        # Step 4: Tier 2 Semantic Cosine Centroid Gating (OOD Detection)
        t2_res = self.tier2_gate.check(contextualized_q, query_vec=q_vec)
        if t2_res.is_ood:
            total_dt = (time.perf_counter() - t0) * 1000.0
            refusal_ans = (
                TIER2_HOMEWORK_REFUSAL_MESSAGE
                if getattr(t2_res, "category", "") == "OUT_OF_SCOPE_HOMEWORK"
                else TIER2_REFUSAL_MESSAGE
            )
            return EngineResult(
                answer=refusal_ans,
                architecture="UnifiedChatEngine (Tier-2 Blocked)",
                retrieved_chunks=[],
                retrieval_latency_ms=0.0,
                generation_latency_ms=0.0,
                total_latency_ms=total_dt,
                tokens_generated=len(refusal_ans.split()),
                routing_mode="GUARDED",
                metadata={
                    "tier": "TIER_2",
                    "category": getattr(t2_res, "category", "OUT_OF_SCOPE"),
                    "academic_score": t2_res.academic_score,
                    "ood_score": t2_res.ood_score,
                    "task_score": getattr(t2_res, "task_score", 0.0),
                },
            )

        # Step 5: Assemble Context & Profile
        sliding_history, current_profile = self.memory.get_context_for_llm(session_id)
        if history and len(sliding_history) < len(history):
            sliding_history = history[-self.memory.max_window_turns * 2 :]

        # Step 6: System-1 Decider Semantic Routing
        dec_res = self.decider.decide(contextualized_q, query_vec=q_vec)
        is_pure_chitchat = (
            dec_res.category == DecisionCategory.CHAT_ONLY
            and dec_res.confidence >= 0.50
            and not (history or sliding_history)
        )

        if is_pure_chitchat:
            t_gen_start = time.perf_counter()
            chitchat_prompt = (
                "Anda adalah Compass AI, asisten akademik pintar Fakultas Teknologi Maju dan Multidisiplin (FTMM) "
                "Universitas Airlangga. Berikan sapaan ramah, padat, dan tawarkan bantuan seputar kurikulum, SKS, "
                "prasyarat matkul, atau perencanaan studi."
            )
            messages = [
                {"role": "system", "content": chitchat_prompt},
                {"role": "user", "content": q_raw},
            ]
            resp = get_ollama_client().chat(
                model=self.model,
                messages=messages,
                options={"temperature": 0.3, "num_predict": 256, "num_ctx": DEFAULT_NUM_CTX},
                keep_alive=-1,
            )
            gen_dt = (time.perf_counter() - t_gen_start) * 1000.0
            total_dt = (time.perf_counter() - t0) * 1000.0
            final_reply = resp.message.content.strip()

            self.memory.add_turn(session_id, q_raw, final_reply)

            return EngineResult(
                answer=final_reply,
                architecture="UnifiedChatEngine (System-1 Direct)",
                retrieved_chunks=[],
                retrieval_latency_ms=0.0,
                generation_latency_ms=gen_dt,
                total_latency_ms=total_dt,
                tokens_generated=resp.get("eval_count", 0),
                routing_mode="DIRECT",
                metadata={"decider_category": dec_res.category.value, "confidence": dec_res.confidence},
            )

        # Step 7: Regulation & SOP FAQ (CAG Route - 0ms retrieval)
        if dec_res.category == DecisionCategory.REGULATION_SOP:
            t_gen_start = time.perf_counter()
            cag_prompt = (
                f"{get_cag_system_prompt()}\n\n"
                "## PERAN SAAT INI (CAG REGULATION FAQ):\n"
                "Jawab pertanyaan mahasiswa mengenai regulasi akademik atau SOP administrasi FTMM secara tepat, "
                "padat, dan lugas sesuai basis pengetahuan resmi di atas. Jangan berhalusinasi. "
                "Sebutkan angka SKS atau ketentuan SOP secara eksplisit (contoh: PKL minimal 80 SKS kumulatif)."
            )
            messages = [{"role": "system", "content": cag_prompt}]
            if sliding_history:
                messages.extend(sliding_history[-4:])
            messages.append({"role": "user", "content": q_raw})

            resp = get_ollama_client().chat(
                model=self.model,
                messages=messages,
                options={"temperature": 0.0, "num_predict": 300, "num_ctx": DEFAULT_NUM_CTX},
                keep_alive=-1,
            )
            gen_dt = (time.perf_counter() - t_gen_start) * 1000.0
            total_dt = (time.perf_counter() - t0) * 1000.0
            final_reply = resp.message.content.strip()

            self.memory.add_turn(session_id, q_raw, final_reply)
            _, updated_prof = self.memory.get_context_for_llm(session_id)

            return EngineResult(
                answer=final_reply,
                architecture="UnifiedChatEngine (CAG Route)",
                retrieved_chunks=[],
                retrieval_latency_ms=0.0,
                generation_latency_ms=gen_dt,
                total_latency_ms=total_dt,
                tokens_generated=resp.get("eval_count", 0),
                routing_mode="CAG",
                metadata={"category": "REGULATION_SOP", "updated_profile": updated_prof},
            )

        # Step 8: ReAct Agent Engine (Native Tool-Calling / Hybrid / RAG)
        react_res = self.react_engine.generate_answer(
            query=contextualized_q,
            history=sliding_history,
            student_profile=current_profile,
            precomputed_decision=dec_res,
        )

        tools_executed = react_res.metadata.get("tools_executed", [])
        tool_names = [t["tool"] for t in tools_executed]

        if (
            "get_courses_by_semester" in tool_names
            or "search_courses" in tool_names
        ) and "generate_degree_plan" not in tool_names:
            routing_mode = "RAG"
        elif (
            dec_res.category == DecisionCategory.DEGREE_PLANNING
            or "generate_degree_plan" in tool_names
        ):
            routing_mode = "HYBRID"
        elif (
            dec_res.category == DecisionCategory.REGULATION_SOP
            or "calculate_sks_limit" in tool_names
        ):
            routing_mode = "CAG"
        elif dec_res.category in (DecisionCategory.COURSE_DISCOVERY, DecisionCategory.PREREQUISITE):
            routing_mode = "RAG"
        elif not tools_executed:
            routing_mode = "DIRECT"
        else:
            routing_mode = "REACT_AGENT"

        self.memory.add_turn(
            session_id,
            q_raw,
            react_res.answer,
            query_vec=q_vec,
            decider_category=dec_res.category.value,
        )

        # Update profile updates into metadata
        _, updated_prof = self.memory.get_context_for_llm(session_id)
        react_res.routing_mode = routing_mode
        react_res.metadata["updated_profile"] = updated_prof
        react_res.metadata["contextualized_query"] = contextualized_q

        react_res.metadata["normalized_query"] = q_clean
        react_res.metadata["normalizer_latency_ms"] = norm_res.latency_ms
        if norm_res.traces:
            react_res.metadata["normalizer_traces"] = [
                {"original": t.original, "normalized": t.normalized, "rule_type": t.rule_type, "distance": t.distance}
                for t in norm_res.traces
            ]
        return react_res
