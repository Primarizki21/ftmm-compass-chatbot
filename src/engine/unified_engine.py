import time
import re
from typing import Any
import ollama

from src.engine.base import EngineResult
from src.guardrails.tier1_filter import Tier1DeterministicFilter
from src.guardrails.tier2_semantic_gate import Tier2SemanticGate
from src.engine.decider import System1Decider, DecisionCategory
from src.engine.memory import ConversationMemoryManager
from src.engine.react_agent import ReActAgentEngine
from src.knowledge.cag_store import get_cag_system_prompt

DEFAULT_MODEL = "qwen2.5:3b"

TIER1_REFUSAL_MESSAGE = (
    "Permintaan tidak dapat diproses karena melanggar kebijakan keamanan sistem akademik FTMM UNAIR."
)

TIER2_REFUSAL_MESSAGE = (
    "Maaf, sebagai asisten akademik FTMM UNAIR, saya hanya dapat membantu pertanyaan terkait "
    "perkuliahan, kurikulum, silabus mata kuliah, regulasi akademik (SKS, PKL, Skripsi, Yudisium), "
    "dan perencanaan studi di FTMM."
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
    ):
        self.model = model
        self.memory = memory_manager or ConversationMemoryManager()
        self.tier1_filter = Tier1DeterministicFilter()
        self.tier2_gate = Tier2SemanticGate()
        self.tier2_gate.initialize()
        self.decider = System1Decider()
        self.decider.initialize()
        self.react_engine = ReActAgentEngine(model=self.model)

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

        # Step 2: Session State Sync & Multi-Turn Contextualization
        session = self.memory.get_or_create_session(session_id)
        if student_profile:
            for k, v in student_profile.items():
                if v is not None and v != [] and not session.student_profile.get(k):
                    session.student_profile[k] = v

        contextualized_q = self.memory.contextualize_query(session_id, q_raw)

        # Step 3: Tier 2 Semantic Cosine Centroid Gating (OOD Detection)
        t2_res = self.tier2_gate.check(contextualized_q)
        if t2_res.is_ood:
            total_dt = (time.perf_counter() - t0) * 1000.0
            return EngineResult(
                answer=TIER2_REFUSAL_MESSAGE,
                architecture="UnifiedChatEngine (Tier-2 Blocked)",
                retrieved_chunks=[],
                retrieval_latency_ms=0.0,
                generation_latency_ms=0.0,
                total_latency_ms=total_dt,
                tokens_generated=len(TIER2_REFUSAL_MESSAGE.split()),
                routing_mode="GUARDED",
                metadata={"tier": "TIER_2", "academic_score": t2_res.academic_score, "ood_score": t2_res.ood_score},
            )

        # Step 4: Assemble Context & Profile
        sliding_history, current_profile = self.memory.get_context_for_llm(session_id)
        if history and len(sliding_history) < len(history):
            sliding_history = history[-self.memory.max_window_turns * 2 :]

        # Step 5: System-1 Decider Fast-Path Routing
        dec_res = self.decider.decide(contextualized_q)
        greeting_match = re.search(
            r"^(halo|hai|hi|hello|selamat pagi|selamat siang|selamat sore|selamat malam|assalamualaikum|tes|ping|terima kasih|makasih|thanks)\b",
            q_raw.strip(),
            re.IGNORECASE,
        )
        is_pure_chitchat = (
            dec_res.category == DecisionCategory.CHAT_ONLY
            and dec_res.confidence >= 0.70
            and bool(greeting_match)
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
            resp = ollama.chat(
                model=self.model,
                messages=messages,
                options={"temperature": 0.3, "num_predict": 256},
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

        # Step 6: Regulation & SOP FAQ (CAG Route - 0ms retrieval)
        is_sop_regulation = bool(
            re.search(
                r"\b(pkl|skripsi|yudisium|cuti|ukt|skma|syarat\s+sks|batas\s+sks|beban\s+sks|ips|ipk)\b",
                contextualized_q,
                re.IGNORECASE,
            )
        )
        is_planning = bool(
            re.search(
                r"\b(rekomendasi|rekomendasikan|rencana\s+studi|degree\s+plan|plotting\s+krs|ambil\s+matkul\s+apa)\b",
                contextualized_q,
                re.IGNORECASE,
            )
        )

        if is_sop_regulation and not is_planning:
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

            resp = ollama.chat(
                model=self.model,
                messages=messages,
                options={"temperature": 0.0, "num_predict": 300},
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

        # Step 7: Native Tool-Calling via ReActAgentEngine

        # Step 6: Native Tool-Calling via ReActAgentEngine
        react_res = self.react_engine.generate_answer(
            query=contextualized_q,
            history=sliding_history,
            student_profile=current_profile,
        )

        # Determine mapped routing mode & action type for frontend contract
        tools_executed = react_res.metadata.get("tools_executed", [])
        tool_names = [t["tool"] for t in tools_executed]

        is_planning = bool(
            re.search(
                r"\b(rekomendasi|rekomendasikan|rencana\s+studi|degree\s+plan|plotting\s+krs|ambil\s+matkul\s+apa)\b",
                contextualized_q,
                re.IGNORECASE,
            )
            or "generate_degree_plan" in tool_names
        )
        is_sop_regulation = bool(
            re.search(
                r"\b(pkl|skripsi|yudisium|cuti|ukt|skma|syarat\s+sks|batas\s+sks|beban\s+sks|ips|ipk)\b",
                contextualized_q,
                re.IGNORECASE,
            )
            or "calculate_sks_limit" in tool_names
        )

        if is_planning:
            routing_mode = "HYBRID"
        elif is_sop_regulation:
            routing_mode = "CAG"
        elif "get_courses_by_semester" in tool_names or "search_courses" in tool_names:
            routing_mode = "RAG"
        elif not tools_executed:
            routing_mode = "DIRECT"
        else:
            routing_mode = "REACT_AGENT"

        # Update memory state
        self.memory.add_turn(session_id, q_raw, react_res.answer)

        # Update profile updates into metadata
        _, updated_prof = self.memory.get_context_for_llm(session_id)
        react_res.routing_mode = routing_mode
        react_res.metadata["updated_profile"] = updated_prof
        react_res.metadata["contextualized_query"] = contextualized_q

        return react_res
