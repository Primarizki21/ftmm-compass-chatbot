import json
import os
import time
from typing import Any
import ollama
from src.engine.base import EngineResult
from src.engine.decider import System1Decider, SEMESTER_COURSE_FAST_PATH_REGEX
from src.engine.tools import ACADEMIC_TOOL_DEFINITIONS, TOOL_REGISTRY
from src.engine.tools.academic_tools import normalize_program_studi, extract_semester_from_text
DEFAULT_MODEL = "qwen2.5:3b"


def _get_ollama_client() -> ollama.Client:
    host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").strip()
    if not host.startswith(("http://", "https://")):
        host = f"http://{host}"
    return ollama.Client(host=host.rstrip("/"))
REACT_SYSTEM_PROMPT = """Anda adalah Compass AI, agen penasihat akademik resmi Fakultas Teknologi Maju dan Multidisiplin (FTMM) Universitas Airlangga.
Tugas Anda adalah membantu mahasiswa dengan informasi akurat, perencanaan studi, dan pengecekan regulasi akademik.
Prinsip Kerja Agen:
1. Gunakan tools deterministik yang tersedia untuk mencari mata kuliah, memvalidasi prasyarat, menghitung batas SKS, dan menyusun rencana studi.
2. Dilarang menebak atau berhalusinasi tentang prasyarat atau batas SKS; selalu rujuk data hasil tools.
3. ATURAN BAHASA (LANGUAGE MIRRORING):
   - Jika pengguna bertanya dalam Bahasa Inggris (English), berikan seluruh respons dalam Bahasa Inggris profesional.
   - Jika pengguna bertanya dalam Bahasa Indonesia, berikan seluruh respons dalam Bahasa Indonesia.
   - Jangan mencampuradukkan bahasa (no code-switching dalam satu kalimat).
4. Jika mahasiswa meminta rekomendasi/penyusunan rencana studi (Degree Plan), Anda WAJIB memanggil fungsi `generate_degree_plan`.
5. Aturan Putusan Prasyarat (Symmetric Decision Rules):
   - Jika 'is_eligible: True' atau 'has_prerequisites: False':
     * Bahasa Indonesia: Kalimat pertama WAJIB: 'Ya, Anda memenuhi syarat dan diperbolehkan mengambil [Nama Mata Kuliah].'
     * English: First sentence MUST be: 'Yes, you meet the prerequisites and are eligible to enroll in [Course Name].'
   - Jika 'is_eligible: False':
     * Bahasa Indonesia: Kalimat pertama WAJIB: 'Belum bisa, Anda belum memenuhi prasyarat untuk mengambil [Nama Mata Kuliah].'
     * English: First sentence MUST be: 'Not yet, you have not met the prerequisites to enroll in [Course Name].'
6. Jangan pernah mengembalikan respons kosong. Selalu berikan jawaban terperinci setelah tools selesai dieksekusi.
7. Jika mahasiswa menanyakan daftar mata kuliah pada semester tertentu (misal: 'mata kuliah semester 3 apa saja?'), Anda WAJIB memanggil fungsi `get_courses_by_semester` (BUKAN `search_courses`).
8. Ketika menampilkan hasil daftar mata kuliah dari `get_courses_by_semester`:
   - Anda WAJIB menampilkan SELURUH mata kuliah yang ada di hasil tool secara lengkap tanpa terpotong (jangan hanya 1 atau sebagian).
   - Format ringkas per baris: Nomor. [Kode MK] Nama MK (X SKS, Jenis) - Prasyarat: [Prasyarat jika ada].
   - Tuliskan juga total mata kuliah dan total SKS sesuai data tool.
9. BATASAN DOMAIN & REFUSAL POLICY (OUT-OF-SCOPE REFUSAL):
   - Anda adalah asisten akademik institusional FTMM UNAIR, BUKAN asisten umum/coding umum.
   - Jika pengguna meminta penulisan kode umum (seperti implementasi algoritma pemrograman lepas, pembuatan script/aplikasi di luar kurikulum FTMM), pengerjaan PR umum, atau topik non-akademik umum lainnya, Anda WAJIB MENOLAK secara sopan dan lugas:
     * Bahasa Indonesia: 'Maaf, sebagai asisten akademik FTMM UNAIR, saya hanya melayani konsultasi kurikulum, prasyarat mata kuliah, beban SKS, dan perencanaan studi FTMM. Saya tidak dapat membuatkan kode program atau mengerjakan tugas di luar kurikulum resmi.'
     * English: 'Sorry, as the official FTMM UNAIR academic assistant, I only handle inquiries related to the curriculum, course prerequisites, credit limits, and degree planning. I cannot write arbitrary code or complete general homework assignments outside our curriculum.'
"""


_decider_instance: System1Decider | None = None


def get_system1_decider() -> System1Decider:
    global _decider_instance
    if _decider_instance is None:
        mrl_dim_env = os.getenv("DECIDER_MRL_DIM", "256")
        mrl_dim = int(mrl_dim_env) if mrl_dim_env and mrl_dim_env.lower() != "none" else None
        _decider_instance = System1Decider(backend="gemma2", mrl_dim=mrl_dim)
        _decider_instance.initialize()
    return _decider_instance


def select_relevant_tools(query: str) -> tuple[list[dict[str, Any]] | None, dict[str, Any]]:
    decider = get_system1_decider()
    res = decider.decide(query)
    telemetry = {
        "decider_category": res.category.value,
        "decider_confidence": res.confidence,
        "decider_latency_ms": res.latency_ms,
    }
    return res.selected_tools, telemetry


class ReActAgentEngine:
    def __init__(self, model: str = DEFAULT_MODEL, max_steps: int = 5):
        self.model = model
        self.max_steps = max_steps

    def generate_answer(
        self,
        query: str,
        history: list[dict[str, str]] | None = None,
        student_profile: dict[str, Any] | None = None,
    ) -> EngineResult:
        t0 = time.perf_counter()
        active_tools, decider_telemetry = select_relevant_tools(query)
        # Build initial message stack
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": REACT_SYSTEM_PROMPT}
        ]
        
        if student_profile:
            profile_context = (
                f"[PROFIL MAHASISWA]:\n"
                f"- Program Studi: {student_profile.get('program_studi', 'Belum ditentukan')}\n"
                f"- Semester Saat Ini: {student_profile.get('semester_saat_ini', 1)}\n"
                f"- Mata Kuliah Lulus: {', '.join(student_profile.get('riwayat_matkul_lulus', [])) or 'Belum ada data'}\n"
                f"- Minat/Fokus: {', '.join(student_profile.get('minat_fokus', [])) or 'Umum'}\n"
                f"- Target Kelulusan: Semester {student_profile.get('target_kelulusan_semester', 8)}\n"
            )
            messages.append({"role": "system", "content": profile_context})

        if history:
            messages.extend(history[-4:])

        messages.append({"role": "user", "content": query})

        total_tool_latency_ms = 0.0
        total_gen_latency_ms = 0.0
        tools_executed: list[dict[str, Any]] = []
        raw_execution_traces: list[dict[str, Any]] = []
        plan_payload: dict[str, Any] | None = None
        action_type = "CHAT"
        step_count = 0
        final_answer = ""
        total_prompt_tokens = 0
        total_completion_tokens = 0
        self_correction_triggered = False
        # Reasoning Loop
        while step_count < self.max_steps:
            step_count += 1
            t_gen_start = time.perf_counter()
            resp = None
            for attempt in range(2):
                try:
                    chat_kwargs: dict[str, Any] = {
                        "model": self.model,
                        "messages": messages,
                        "options": {"temperature": 0.1, "top_p": 0.9, "num_predict": 1024, "num_ctx": 4096},
                    }
                    if active_tools:
                        chat_kwargs["tools"] = active_tools
                    client = _get_ollama_client()
                    resp = client.chat(**chat_kwargs)
                    break
                except Exception as e:
                    if attempt == 0:
                        time.sleep(0.6)
                        continue
                    raise e
            gen_time = (time.perf_counter() - t_gen_start) * 1000.0
            total_gen_latency_ms += gen_time
            p_tokens = resp.get("prompt_eval_count", 0)
            c_tokens = resp.get("eval_count", 0)
            total_prompt_tokens += p_tokens
            total_completion_tokens += c_tokens
            msg = resp.message
            tool_calls = getattr(msg, "tool_calls", None)
            if not tool_calls and step_count == 1 and not tools_executed and SEMESTER_COURSE_FAST_PATH_REGEX.search(query):
                detected_p = normalize_program_studi(query)
                detected_s = extract_semester_from_text(query)
                if detected_p != query and detected_s is not None:
                    class _Fn:
                        name = "get_courses_by_semester"
                        arguments = {"program_studi": detected_p, "semester": detected_s}
                    class _TC:
                        function = _Fn()
                    tool_calls = [_TC()]

            # Case A: Model outputs tool calls -> execute tools
            if tool_calls:
                messages.append(msg)
                step_tools_record = []
                for tc in tool_calls:
                    fn_name = tc.function.name
                    args = tc.function.arguments
                    
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except Exception:
                            args = {}
                            
                    # Auto-inject missing profile attributes
                    if student_profile:
                        if fn_name == "generate_degree_plan":
                            if not args.get("program_studi"):
                                args["program_studi"] = student_profile.get("program_studi", "Teknologi Sains Data")
                            if not args.get("passed_courses"):
                                args["passed_courses"] = student_profile.get("riwayat_matkul_lulus", [])
                            if not args.get("target_semester"):
                                args["target_semester"] = student_profile.get("target_kelulusan_semester", 8)
                            if not args.get("focus_tracks"):
                                args["focus_tracks"] = student_profile.get("minat_fokus", [])
                        elif fn_name == "validate_prerequisites_dag":
                            if "passed_courses" not in args or not args.get("passed_courses"):
                                args["passed_courses"] = student_profile.get("riwayat_matkul_lulus", [])

                    if fn_name == "get_courses_by_semester":
                        prodi_from_q = normalize_program_studi(query)
                        if prodi_from_q != query:
                            args["program_studi"] = prodi_from_q
                        elif not args.get("program_studi") and student_profile and student_profile.get("program_studi"):
                            args["program_studi"] = student_profile.get("program_studi")

                        sem_from_q = extract_semester_from_text(query)
                        if sem_from_q is not None:
                            args["semester"] = sem_from_q
                        elif not args.get("semester") and student_profile and student_profile.get("semester_saat_ini"):
                            args["semester"] = student_profile.get("semester_saat_ini")
                    t_tool_start = time.perf_counter()
                    if fn_name in TOOL_REGISTRY:
                        tool_fn = TOOL_REGISTRY[fn_name]
                        try:
                            result = tool_fn(**args)
                        except Exception as e:
                            result = {
                                "status": "error",
                                "error": str(e),
                                "hint": "Perbaiki argumen fungsi sesuai schema parameter."
                            }
                            self_correction_triggered = True
                    else:
                        result = {"status": "error", "error": f"Tool '{fn_name}' tidak terdaftar."}
                        self_correction_triggered = True
                    tool_time = (time.perf_counter() - t_tool_start) * 1000.0
                    total_tool_latency_ms += tool_time

                    tools_executed.append({
                        "step": step_count,
                        "tool": fn_name,
                        "args": args,
                        "latency_ms": tool_time,
                    })

                    # If degree plan was generated, capture payload for frontend
                    if fn_name == "generate_degree_plan" and isinstance(result, dict) and "action" in result:
                        plan_payload = result
                        action_type = "PLAN_GENERATED"
                    step_tools_record.append({
                        "tool": fn_name,
                        "args_raw": args,
                        "result_raw": result,
                        "latency_ms": round(tool_time, 2)
                    })

                    messages.append({
                        "role": "tool",
                        "content": json.dumps(result, ensure_ascii=False)
                    })

                raw_execution_traces.append({
                    "step": step_count,
                    "prompt_tokens": p_tokens,
                    "completion_tokens": c_tokens,
                    "gen_latency_ms": round(gen_time, 2),
                    "action": "TOOL_CALL",
                    "tools": step_tools_record,
                })
            else:
                # Case B: Model completed reasoning and returned final text
                final_answer = msg.content.strip()
                raw_execution_traces.append({
                    "step": step_count,
                    "prompt_tokens": p_tokens,
                    "completion_tokens": c_tokens,
                    "gen_latency_ms": round(gen_time, 2),
                    "action": "FINAL_ANSWER",
                    "answer": final_answer,
                })
                break
        if not final_answer and messages:
            last_content = getattr(messages[-1], "content", "")
            if isinstance(last_content, str):
                final_answer = last_content

        total_latency_ms = (time.perf_counter() - t0) * 1000.0

        return EngineResult(
            answer=final_answer,
            architecture="ReAct Agent (Native Tool-Calling)",
            retrieved_chunks=[f"Tool: {t['tool']} -> {t['args']}" for t in tools_executed],
            retrieval_latency_ms=total_tool_latency_ms,
            generation_latency_ms=total_gen_latency_ms,
            total_latency_ms=total_latency_ms,
            tokens_generated=total_completion_tokens,
            routing_mode="REACT_AGENT",
            metadata={
                "steps": step_count,
                "action_type": action_type,
                "plan_payload": plan_payload,
                "tools_executed": tools_executed,
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "self_correction_triggered": self_correction_triggered,
                "execution_traces": raw_execution_traces,
                **decider_telemetry,
            }
        )
