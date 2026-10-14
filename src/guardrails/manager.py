import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.engine.base import EngineResult
from src.engine.react_agent import ReActAgentEngine
from .tier1_filter import Tier1DeterministicFilter
from .tier2_semantic_gate import Tier2SemanticGate
from .tier3_nemo_rails import Tier3NeMoRails
from .tier4_tool_rail import Tier4ToolRail


class GuardVariant(str, Enum):
    VARIANT_A_BASELINE = "VARIANT_A_BASELINE"
    VARIANT_B_NEMO = "VARIANT_B_NEMO"
    VARIANT_C_LIGHTWEIGHT = "VARIANT_C_LIGHTWEIGHT"
    VARIANT_D_HYBRID = "VARIANT_D_HYBRID"


@dataclass
class GuardrailsResult:
    query: str
    variant: GuardVariant
    response: str
    blocked: bool
    trigger_tier: str | None
    refusal_reason: str | None
    latency_total_ms: float
    tier_latencies_ms: dict[str, float] = field(default_factory=dict)
    tools_executed: list[dict[str, Any]] = field(default_factory=list)
    action_type: str = "ANSWER"
    plan_payload: dict[str, Any] | None = None


REFUSAL_MESSAGES = {
    "TIER_1": "Permintaan tidak dapat diproses karena melanggar kebijakan keamanan sistem akademik FTMM UNAIR.",
    "TIER_2": "Maaf, sebagai asisten akademik FTMM UNAIR, saya hanya dapat membantu pertanyaan terkait perkuliahan, kurikulum, dan regulasi studi.",
    "TIER_2_HOMEWORK": "Maaf, sebagai asisten akademik FTMM UNAIR, saya berfokus pada informasi kurikulum, silabus mata kuliah, prasyarat, dan perencanaan studi. Saya tidak dapat mengerjakan tugas/soal, menurunkan rumus, atau menuliskan kode program mandiri. Silakan konsultasikan topik materi terkait dengan silabus mata kuliah resmi FTMM.",
    "TIER_3": "Maaf, permintaan ini melanggar kebijakan batasan topik atau panduan integritas akademik FTMM UNAIR.",
    "TIER_4": "Operasi dibatalkan: parameter yang diberikan melanggar batas legal regulasi akademik FTMM UNAIR.",
}


class SafetyGuardrailsManager:
    """Master Manager for Academic Safety Guardrails.

    Orchestrates Tier 1 (Deterministic Filter), Tier 2 (Semantic Gate),
    Tier 3 (NeMo Rails), and Tier 4 (Tool Rail) across 4 experimental variants.
    """

    def __init__(self, agent_engine: ReActAgentEngine | None = None):
        self.agent = agent_engine or ReActAgentEngine()
        self.tier1 = Tier1DeterministicFilter()
        self.tier2 = Tier2SemanticGate()
        self.tier3 = Tier3NeMoRails()
        self.tier4 = Tier4ToolRail()

    def process_turn(
        self,
        query: str,
        variant: GuardVariant = GuardVariant.VARIANT_C_LIGHTWEIGHT,
        history: list[dict[str, str]] | None = None,
        student_profile: dict[str, Any] | None = None,
    ) -> GuardrailsResult:
        t_start = time.perf_counter()
        tier_lats: dict[str, float] = {}

        # -----------------------------------------------------------
        # VARIANT A: Baseline (Raw ReAct Agent without guardrails)
        # -----------------------------------------------------------
        if variant == GuardVariant.VARIANT_A_BASELINE:
            agent_res: EngineResult = self.agent.generate_answer(
                query=query,
                history=history,
                student_profile=student_profile,
            )
            total_dt = (time.perf_counter() - t_start) * 1000

            refusal_markers = ["tidak dapat", "kebijakan", "maaf, saya hanya", "i cannot"]
            blocked = any(m in agent_res.answer.lower() for m in refusal_markers)
            tools_executed = agent_res.metadata.get("tools_executed", [])
            action_type = agent_res.metadata.get("action_type", "ANSWER")
            plan_payload = agent_res.metadata.get("plan_payload", None)

            return GuardrailsResult(
                query=query,
                variant=variant,
                response=agent_res.answer,
                blocked=blocked,
                trigger_tier=None,
                refusal_reason="Spontaneous LLM refusal" if blocked else None,
                latency_total_ms=total_dt,
                tier_latencies_ms={"agent_ms": agent_res.retrieval_latency_ms},
                tools_executed=tools_executed,
                action_type=action_type,
                plan_payload=plan_payload,
            )

        # -----------------------------------------------------------
        # VARIANT B: NeMo Guardrails alone
        # -----------------------------------------------------------
        if variant == GuardVariant.VARIANT_B_NEMO:
            t0_nemo = time.perf_counter()
            nemo_res = self.tier3.check(query)
            dt_nemo = (time.perf_counter() - t0_nemo) * 1000
            tier_lats["tier3_nemo_ms"] = dt_nemo
            total_dt = (time.perf_counter() - t_start) * 1000

            return GuardrailsResult(
                query=query,
                variant=variant,
                response=nemo_res.response,
                blocked=nemo_res.blocked,
                trigger_tier="TIER_3" if nemo_res.blocked else None,
                refusal_reason=nemo_res.reason,
                latency_total_ms=total_dt,
                tier_latencies_ms=tier_lats,
                action_type="GUARDED_REFUSAL" if nemo_res.blocked else "ANSWER",
            )

        # -----------------------------------------------------------
        # VARIANT C: Lightweight Cascaded Guard (Tier 1 + Tier 2 + Tier 4)
        # -----------------------------------------------------------
        if variant == GuardVariant.VARIANT_C_LIGHTWEIGHT:
            # Tier 1: Deterministic Regex Filter (0ms)
            t0_t1 = time.perf_counter()
            f1 = self.tier1.check(query)
            dt_t1 = (time.perf_counter() - t0_t1) * 1000
            tier_lats["tier1_filter_ms"] = dt_t1

            if f1.blocked:
                total_dt = (time.perf_counter() - t_start) * 1000
                return GuardrailsResult(
                    query=query,
                    variant=variant,
                    response=REFUSAL_MESSAGES["TIER_1"],
                    blocked=True,
                    trigger_tier="TIER_1",
                    refusal_reason=f1.reason,
                    latency_total_ms=total_dt,
                    tier_latencies_ms=tier_lats,
                    action_type="GUARDED_REFUSAL",
                )

            # Tier 2: Embedding Semantic Gating (<5ms)
            t0_t2 = time.perf_counter()
            g2 = self.tier2.check(query)
            dt_t2 = (time.perf_counter() - t0_t2) * 1000
            tier_lats["tier2_semantic_ms"] = dt_t2

            if g2.is_ood:
                total_dt = (time.perf_counter() - t_start) * 1000
                ref_key = "TIER_2_HOMEWORK" if getattr(g2, "category", "") == "OUT_OF_SCOPE_HOMEWORK" else "TIER_2"
                return GuardrailsResult(
                    query=query,
                    variant=variant,
                    response=REFUSAL_MESSAGES[ref_key],
                    blocked=True,
                    trigger_tier="TIER_2",
                    refusal_reason=g2.reason,
                    latency_total_ms=total_dt,
                    tier_latencies_ms=tier_lats,
                    action_type="GUARDED_REFUSAL",
                )

            # Tier 4: Executing Agent with Tool Rail protection
            t0_agent = time.perf_counter()
            agent_res = self.agent.generate_answer(query=query, history=history, student_profile=student_profile)
            dt_agent = (time.perf_counter() - t0_agent) * 1000
            tier_lats["agent_ms"] = dt_agent
            total_dt = (time.perf_counter() - t_start) * 1000

            tools_executed = agent_res.metadata.get("tools_executed", [])
            action_type = agent_res.metadata.get("action_type", "ANSWER")
            plan_payload = agent_res.metadata.get("plan_payload", None)

            sanitized_tools = []
            for tool_call in tools_executed:
                t_name = tool_call.get("tool")
                t_args = tool_call.get("args", {})
                v_res = self.tier4.validate_call(t_name, t_args)
                if not v_res.is_valid:
                    return GuardrailsResult(
                        query=query,
                        variant=variant,
                        response=REFUSAL_MESSAGES["TIER_4"],
                        blocked=True,
                        trigger_tier="TIER_4",
                        refusal_reason=v_res.error,
                        latency_total_ms=total_dt,
                        tier_latencies_ms=tier_lats,
                        action_type="GUARDED_REFUSAL",
                    )
                sanitized_tools.append(tool_call)

            return GuardrailsResult(
                query=query,
                variant=variant,
                response=agent_res.answer,
                blocked=False,
                trigger_tier=None,
                refusal_reason=None,
                latency_total_ms=total_dt,
                tier_latencies_ms=tier_lats,
                tools_executed=sanitized_tools,
                action_type=action_type,
                plan_payload=plan_payload,
            )

        # -----------------------------------------------------------
        # VARIANT D: Hybrid Defense (Tier 1 + Tier 2 + NeMo + Tier 4)
        # -----------------------------------------------------------
        if variant == GuardVariant.VARIANT_D_HYBRID:
            # Tier 1
            t0_t1 = time.perf_counter()
            f1 = self.tier1.check(query)
            dt_t1 = (time.perf_counter() - t0_t1) * 1000
            tier_lats["tier1_filter_ms"] = dt_t1
            if f1.blocked:
                total_dt = (time.perf_counter() - t_start) * 1000
                return GuardrailsResult(
                    query=query,
                    variant=variant,
                    response=REFUSAL_MESSAGES["TIER_1"],
                    blocked=True,
                    trigger_tier="TIER_1",
                    refusal_reason=f1.reason,
                    latency_total_ms=total_dt,
                    tier_latencies_ms=tier_lats,
                    action_type="GUARDED_REFUSAL",
                )

            # Tier 2
            t0_t2 = time.perf_counter()
            g2 = self.tier2.check(query)
            dt_t2 = (time.perf_counter() - t0_t2) * 1000
            tier_lats["tier2_semantic_ms"] = dt_t2
            if g2.is_ood:
                total_dt = (time.perf_counter() - t_start) * 1000
                return GuardrailsResult(
                    query=query,
                    variant=variant,
                    response=REFUSAL_MESSAGES["TIER_2_HOMEWORK" if getattr(g2, "category", "") == "OUT_OF_SCOPE_HOMEWORK" else "TIER_2"],
                    blocked=True,
                    trigger_tier="TIER_2",
                    refusal_reason=g2.reason,
                    latency_total_ms=total_dt,
                    tier_latencies_ms=tier_lats,
                    action_type="GUARDED_REFUSAL",
                )

            # Tier 3 (NeMo Semantic Safety Layer)
            t0_t3 = time.perf_counter()
            n3 = self.tier3.check(query)
            dt_t3 = (time.perf_counter() - t0_t3) * 1000
            tier_lats["tier3_nemo_ms"] = dt_t3
            if n3.blocked:
                total_dt = (time.perf_counter() - t_start) * 1000
                return GuardrailsResult(
                    query=query,
                    variant=variant,
                    response=n3.response,
                    blocked=True,
                    trigger_tier="TIER_3",
                    refusal_reason=n3.reason,
                    latency_total_ms=total_dt,
                    tier_latencies_ms=tier_lats,
                    action_type="GUARDED_REFUSAL",
                )

            # Agent & Tier 4 (Tool Rail)
            t0_agent = time.perf_counter()
            agent_res = self.agent.generate_answer(query=query, history=history, student_profile=student_profile)
            dt_agent = (time.perf_counter() - t0_agent) * 1000
            tier_lats["agent_ms"] = dt_agent
            total_dt = (time.perf_counter() - t_start) * 1000

            tools_executed = agent_res.metadata.get("tools_executed", [])
            action_type = agent_res.metadata.get("action_type", "ANSWER")
            plan_payload = agent_res.metadata.get("plan_payload", None)

            sanitized_tools = []
            for tool_call in tools_executed:
                t_name = tool_call.get("tool")
                t_args = tool_call.get("args", {})
                v_res = self.tier4.validate_call(t_name, t_args)
                if not v_res.is_valid:
                    return GuardrailsResult(
                        query=query,
                        variant=variant,
                        response=REFUSAL_MESSAGES["TIER_4"],
                        blocked=True,
                        trigger_tier="TIER_4",
                        refusal_reason=v_res.error,
                        latency_total_ms=total_dt,
                        tier_latencies_ms=tier_lats,
                        action_type="GUARDED_REFUSAL",
                    )
                sanitized_tools.append(tool_call)

            return GuardrailsResult(
                query=query,
                variant=variant,
                response=agent_res.answer,
                blocked=False,
                trigger_tier=None,
                refusal_reason=None,
                latency_total_ms=total_dt,
                tier_latencies_ms=tier_lats,
                tools_executed=sanitized_tools,
                action_type=action_type,
                plan_payload=plan_payload,
            )

        raise ValueError(f"Unknown guard variant: {variant}")
