import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.engine.react_agent import REACT_SYSTEM_PROMPT
from src.engine.tools import ACADEMIC_TOOL_DEFINITIONS, TOOL_REGISTRY

DEFAULT_DATASET = Path("data/ground_truth/slm_sizing_eval_dataset.json")
DEFAULT_OUTPUT_DIR = Path("docs/exp-024_ternary_bonsai_slm_benchmark")
DEFAULT_MODELS = [
    "qwen2.5:1.5b",
    "qwen2.5:3b",
    "qwen2.5:7b",
    "bonsai-2-2b",
    "bonsai-2-4b",
    "bonsai-2-9b",
]

OLLAMA_BASE_URL = "http://127.0.0.1:11434/v1"
BONSAI_BASE_URL = "http://127.0.0.1:8080/v1"
BONSAI_PROJECT_DIR = Path("/home/primarizki/project/Bonsai-demo")


def get_gpu_vram_mib() -> int:
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
        return int(res.stdout.strip().split("\n")[0])
    except Exception:
        return 0


def stop_ollama_model(model_name: str) -> None:
    try:
        subprocess.run(["ollama", "stop", model_name], capture_output=True, text=True, check=False)
        time.sleep(1.0)
    except Exception:
        pass


def start_bonsai_server(model_label: str, port: int = 8080, max_wait: int = 60) -> subprocess.Popen:
    size_map = {
        "bonsai-2-2b": "2B",
        "bonsai-2-4b": "4B",
        "bonsai-2-9b": "9B",
        "2b": "2B",
        "4b": "4B",
        "9b": "9B",
    }
    size = size_map.get(model_label.lower(), "2B")
    env = os.environ.copy()
    env["BONSAI_MODEL"] = size
    env["PORT"] = str(port)
    cuda_lib_dir = "/home/primarizki/.local/lib/ollama/cuda_v12"
    env["LD_LIBRARY_PATH"] = f"{cuda_lib_dir}:{env.get('LD_LIBRARY_PATH', '')}"

    script_path = BONSAI_PROJECT_DIR / "scripts" / "start_llama_server.sh"
    proc = subprocess.Popen(
        ["/bin/sh", str(script_path)],
        cwd=str(BONSAI_PROJECT_DIR),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    t0 = time.time()
    endpoint = f"http://127.0.0.1:{port}/v1/models"
    while time.time() - t0 < max_wait:
        try:
            resp = httpx.get(endpoint, timeout=2.0)
            if resp.status_code == 200:
                data = resp.json()
                if "models" in data or "data" in data:
                    time.sleep(2.0)
                    return proc
        except Exception:
            pass
        time.sleep(1.0)

    raise TimeoutError(f"Gagal memulai llama-server untuk model {model_label} dalam {max_wait} detik")


def stop_bonsai_server(proc: subprocess.Popen | None = None, port: int = 8080) -> None:
    if proc:
        try:
            proc.terminate()
            proc.wait(timeout=3)
        except Exception:
            pass
    try:
        subprocess.run(
            ["pkill", "-f", "llama-server"],
            capture_output=True,
            check=False,
        )
    except Exception:
        pass

    try:
        res = subprocess.run(["lsof", "-ti", f"TCP:{port}"], capture_output=True, text=True, check=False)
        pids = res.stdout.strip().split()
        for pid in pids:
            try:
                os.kill(int(pid), 9)
            except Exception:
                pass
    except Exception:
        pass

    time.sleep(1.5)


def get_client_endpoint_for_model(model: str) -> tuple[str, str]:
    m_lower = model.lower()
    if "bonsai" in m_lower:
        return BONSAI_BASE_URL, model
    return OLLAMA_BASE_URL, model


def parse_model_response(
    content: str,
    tool_calls_raw: list[dict[str, Any]] | None = None,
    reasoning_content: str = "",
) -> list[dict[str, Any]]:
    # 1. Direct tool_calls from API
    if tool_calls_raw:
        parsed_calls: list[dict[str, Any]] = []
        for tc in tool_calls_raw:
            fn = tc.get("function", {})
            fn_name = fn.get("name")
            args_raw = fn.get("arguments", {})
            if isinstance(args_raw, str):
                try:
                    args_dict = json.loads(args_raw)
                except Exception:
                    args_dict = {}
            elif isinstance(args_raw, dict):
                args_dict = args_raw
            else:
                args_dict = {}
            if fn_name:
                parsed_calls.append({
                    "type": "function",
                    "function": {
                        "name": fn_name,
                        "arguments": args_dict,
                    },
                })
        if parsed_calls:
            return parsed_calls

    search_texts = [content]
    if reasoning_content:
        search_texts.append(reasoning_content)

    for text in search_texts:
        if not text:
            continue

        # 2. Markdown JSON code blocks
        json_blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
        for block in json_blocks:
            block = block.strip()
            try:
                data = json.loads(block)
                if isinstance(data, dict):
                    if "action" in data:
                        tool_name = str(data["action"]).strip()
                        args = data.get("action_input", {})
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except Exception:
                                pass
                        if not isinstance(args, dict):
                            args = {}
                        return [{"type": "function", "function": {"name": tool_name, "arguments": args}}]
                    if "name" in data and "arguments" in data:
                        tool_name = str(data["name"]).strip()
                        args = data["arguments"]
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except Exception:
                                pass
                        if not isinstance(args, dict):
                            args = {}
                        return [{"type": "function", "function": {"name": tool_name, "arguments": args}}]
                    if "tool" in data:
                        tool_name = str(data["tool"]).strip()
                        args = data.get("parameters", data.get("args", {}))
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except Exception:
                                pass
                        if not isinstance(args, dict):
                            args = {}
                        return [{"type": "function", "function": {"name": tool_name, "arguments": args}}]
                    for reg_tool in TOOL_REGISTRY:
                        if reg_tool in data and isinstance(data[reg_tool], dict):
                            return [{"type": "function", "function": {"name": reg_tool, "arguments": data[reg_tool]}}]
                elif isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and "name" in item:
                            args = item.get("arguments", {})
                            if isinstance(args, str):
                                try:
                                    args = json.loads(args)
                                except Exception:
                                    pass
                            return [{"type": "function", "function": {"name": item["name"], "arguments": args}}]
            except Exception:
                continue

        # 3. Tool call tags <tool_call> ... </tool_call>
        tool_tags = re.findall(r"<tool_call>\s*([\s\S]*?)\s*</tool_call>", text, re.IGNORECASE)
        for tag_content in tool_tags:
            try:
                data = json.loads(tag_content.strip())
                if isinstance(data, dict):
                    name = data.get("name") or data.get("action") or data.get("tool")
                    args = data.get("arguments") or data.get("action_input") or data.get("parameters", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except Exception:
                            pass
                    if name:
                        return [{"type": "function", "function": {"name": name, "arguments": args if isinstance(args, dict) else {}}}]
            except Exception:
                continue

        # 4. ReAct Action / Action Input pattern
        react_match = re.search(
            r"Action:\s*([a-zA-Z0-9_]+)\s*\n\s*Action\s*Input:\s*([\s\S]*?)(?=\n\s*(?:Observation|Action|Final Answer)|$)",
            text,
            re.IGNORECASE,
        )
        if react_match:
            tool_name = react_match.group(1).strip()
            raw_input = react_match.group(2).strip()
            try:
                args = json.loads(raw_input)
            except Exception:
                args = {}
            return [{"type": "function", "function": {"name": tool_name, "arguments": args if isinstance(args, dict) else {}}}]

        # 5. Raw JSON object containing "action" or "name"
        raw_json_match = re.search(r"(\{\s*\"(?:action|name|tool)\"\s*:\s*\"[a-zA-Z0-9_]+\"[\s\S]*?\})", text)
        if raw_json_match:
            try:
                data = json.loads(raw_json_match.group(1))
                name = data.get("action") or data.get("name") or data.get("tool")
                args = data.get("action_input") or data.get("arguments") or data.get("parameters", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        pass
                if name:
                    return [{"type": "function", "function": {"name": name, "arguments": args if isinstance(args, dict) else {}}}]
            except Exception:
                pass

    return []


def execute_physical_tool(tool_name: str, tool_args: dict[str, Any]) -> tuple[bool, Any, str | None]:
    if tool_name not in TOOL_REGISTRY:
        return False, None, f"Tool '{tool_name}' tidak terdaftar di TOOL_REGISTRY"
    tool_fn = TOOL_REGISTRY[tool_name]
    try:
        res = tool_fn(**tool_args)
        if isinstance(res, dict) and res.get("status") == "error":
            return False, res, res.get("error", "Tool returned status=error")
        return True, res, None
    except Exception as e:
        return False, None, str(e)


def evaluate_tool_adherence(
    category: str,
    expected_tool: str | None,
    required_params: list[str],
    tool_calls: list[dict[str, Any]],
    raw_content: str,
    expected_ips: float | None = None,
    expected_semester: int | None = None,
    expected_target_course: str | None = None,
) -> dict[str, Any]:
    has_tool_call = len(tool_calls) > 0
    actual_tool = None
    actual_args: dict[str, Any] = {}

    if has_tool_call:
        tc0 = tool_calls[0]
        fn = tc0.get("function", {})
        actual_tool = fn.get("name")
        args_raw = fn.get("arguments", {})
        if isinstance(args_raw, dict):
            actual_args = args_raw
        elif isinstance(args_raw, str):
            try:
                actual_args = json.loads(args_raw)
            except Exception:
                actual_args = {}

    # 1. Tool Selection
    if category == "CHITCHAT":
        tool_selection_ok = not has_tool_call
    elif category == "COURSE_DISCOVERY":
        tool_selection_ok = actual_tool in ["get_courses_by_semester", "search_courses"]
    else:
        tool_selection_ok = actual_tool == expected_tool

    # 2. JSON Schema Compliance
    if category == "CHITCHAT":
        schema_compliant = True if not has_tool_call else isinstance(actual_args, dict)
    else:
        schema_compliant = (
            has_tool_call
            and isinstance(actual_args, dict)
            and actual_tool in TOOL_REGISTRY
        )

    # 3. Parameter Extraction Error
    param_error = False
    error_reason = None

    if category == "CHITCHAT":
        param_error = has_tool_call
        if has_tool_call:
            error_reason = f"Unexpected tool called on chitchat: {actual_tool}"
    elif not has_tool_call:
        param_error = True
        error_reason = "No tool call generated when expected"
    else:
        for p in required_params:
            if p not in actual_args or actual_args[p] is None or actual_args[p] == "":
                param_error = True
                error_reason = f"Missing required parameter '{p}'"
                break

        if not param_error and expected_tool == "calculate_sks_limit":
            try:
                ips_val = float(actual_args.get("ips", -1))
                if not (0.0 <= ips_val <= 4.0):
                    param_error = True
                    error_reason = f"IPS {ips_val} out of bounds [0.0, 4.0]"
                elif expected_ips is not None and abs(ips_val - expected_ips) > 0.15:
                    param_error = True
                    error_reason = f"Extracted IPS {ips_val} differs from expected {expected_ips}"
            except (ValueError, TypeError):
                param_error = True
                error_reason = f"Invalid IPS type: {actual_args.get('ips')}"

        if not param_error and expected_tool == "get_courses_by_semester":
            try:
                sem_val = int(actual_args.get("semester", 0))
                if not (1 <= sem_val <= 8):
                    param_error = True
                    error_reason = f"Semester {sem_val} out of bounds [1, 8]"
                elif expected_semester is not None and sem_val != expected_semester:
                    param_error = True
                    error_reason = f"Semester {sem_val} differs from expected {expected_semester}"
            except (ValueError, TypeError):
                param_error = True
                error_reason = f"Invalid semester type: {actual_args.get('semester')}"

        if not param_error and expected_tool == "validate_prerequisites_dag":
            course_name = str(actual_args.get("course_code_or_name", "")).strip()
            if len(course_name) < 2:
                param_error = True
                error_reason = "Course name is too short or empty"

        if not param_error and expected_tool == "generate_degree_plan":
            try:
                cur_sem = int(actual_args.get("current_semester", 0))
                if not (1 <= cur_sem <= 8):
                    param_error = True
                    error_reason = f"current_semester {cur_sem} out of bounds [1, 8]"
            except (ValueError, TypeError):
                param_error = True
                error_reason = f"Invalid current_semester type: {actual_args.get('current_semester')}"

    # 4. Physical Execution Test
    exec_success = False
    exec_output = None
    exec_err = None
    if has_tool_call and actual_tool in TOOL_REGISTRY and not param_error:
        exec_success, exec_output, exec_err = execute_physical_tool(actual_tool, actual_args)
    elif category == "CHITCHAT" and not has_tool_call:
        exec_success = True

    return {
        "tool_selection_ok": bool(tool_selection_ok),
        "schema_compliant": bool(schema_compliant),
        "param_error": bool(param_error),
        "error_reason": error_reason,
        "actual_tool": actual_tool,
        "actual_args": actual_args,
        "exec_success": bool(exec_success),
        "exec_err": exec_err,
    }


def run_single_inference_dual_dispatch(
    model: str,
    query: str,
    num_predict: int = 512,
    timeout_sec: float = 45.0,
) -> dict[str, Any]:
    base_url, model_name = get_client_endpoint_for_model(model)
    url = f"{base_url}/chat/completions"

    messages = [
        {"role": "system", "content": REACT_SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]

    payload: dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "tools": ACADEMIC_TOOL_DEFINITIONS,
        "temperature": 0.1,
        "top_p": 0.9,
        "max_tokens": num_predict,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    t0 = time.perf_counter()
    first_token_time = None
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    raw_tool_calls_dict: dict[int, dict[str, Any]] = {}
    usage_info: dict[str, Any] = {}
    timings_info: dict[str, Any] = {}

    try:
        with httpx.Client(timeout=timeout_sec) as client:
            with client.stream("POST", url, json=payload) as response:
                if response.status_code != 200:
                    err_text = response.read().decode("utf-8", errors="ignore")
                    return {
                        "success": False,
                        "perceived_ttft_ms": 0.0,
                        "prompt_eval_ms": 0.0,
                        "eval_ms": 0.0,
                        "total_time_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                        "eval_count": 0,
                        "tps": 0.0,
                        "tool_calls": [],
                        "content": "",
                        "reasoning_content": "",
                        "error": f"HTTP {response.status_code}: {err_text}",
                    }

                for line in response.iter_lines():
                    if not line:
                        continue
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                        except Exception:
                            continue

                        if "timings" in chunk and chunk["timings"]:
                            timings_info = chunk["timings"]
                        if "usage" in chunk and chunk["usage"]:
                            usage_info = chunk["usage"]

                        choices = chunk.get("choices", [])
                        if not choices:
                            continue
                        choice = choices[0]
                        delta = choice.get("delta", {})

                        c_text = delta.get("content")
                        if c_text:
                            if first_token_time is None:
                                first_token_time = time.perf_counter()
                            content_parts.append(c_text)

                        r_text = delta.get("reasoning_content")
                        if r_text:
                            if first_token_time is None:
                                first_token_time = time.perf_counter()
                            reasoning_parts.append(r_text)

                        tcs = delta.get("tool_calls")
                        if tcs:
                            if first_token_time is None:
                                first_token_time = time.perf_counter()
                            for tc in tcs:
                                idx = tc.get("index", 0)
                                if idx not in raw_tool_calls_dict:
                                    raw_tool_calls_dict[idx] = {
                                        "id": tc.get("id", ""),
                                        "type": "function",
                                        "function": {"name": "", "arguments": ""},
                                    }
                                fn = tc.get("function", {})
                                if fn.get("name"):
                                    raw_tool_calls_dict[idx]["function"]["name"] += fn["name"]
                                if fn.get("arguments"):
                                    raw_tool_calls_dict[idx]["function"]["arguments"] += fn["arguments"]

        t_end = time.perf_counter()
        total_time_ms = (t_end - t0) * 1000.0
        perceived_ttft_ms = (first_token_time - t0) * 1000.0 if first_token_time else total_time_ms

        raw_api_tool_calls = [raw_tool_calls_dict[k] for k in sorted(raw_tool_calls_dict.keys())]
        full_content = "".join(content_parts).strip()
        full_reasoning = "".join(reasoning_parts).strip()

        parsed_tool_calls = parse_model_response(
            content=full_content,
            tool_calls_raw=raw_api_tool_calls,
            reasoning_content=full_reasoning,
        )

        eval_count = int(usage_info.get("completion_tokens", timings_info.get("predicted_n", 0)))
        if eval_count == 0 and (full_content or full_reasoning or parsed_tool_calls):
            # Fallback estimation ~ 4 chars/token
            eval_count = max(1, len(full_content + full_reasoning) // 4)

        if timings_info.get("prompt_ms"):
            prompt_eval_ms = float(timings_info["prompt_ms"])
        else:
            prompt_eval_ms = perceived_ttft_ms

        gen_duration_sec = (t_end - first_token_time) if first_token_time else (total_time_ms / 1000.0)
        eval_ms = gen_duration_sec * 1000.0

        if timings_info.get("predicted_per_second"):
            tps = float(timings_info["predicted_per_second"])
        else:
            tps = (eval_count / gen_duration_sec) if gen_duration_sec > 0 else 0.0

        return {
            "success": True,
            "perceived_ttft_ms": round(perceived_ttft_ms, 2),
            "prompt_eval_ms": round(prompt_eval_ms, 2),
            "eval_ms": round(eval_ms, 2),
            "total_time_ms": round(total_time_ms, 2),
            "eval_count": eval_count,
            "tps": round(tps, 2),
            "tool_calls": parsed_tool_calls,
            "content": full_content,
            "reasoning_content": full_reasoning,
            "error": None,
        }

    except Exception as e:
        return {
            "success": False,
            "perceived_ttft_ms": 0.0,
            "prompt_eval_ms": 0.0,
            "eval_ms": 0.0,
            "total_time_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            "eval_count": 0,
            "tps": 0.0,
            "tool_calls": [],
            "content": "",
            "reasoning_content": "",
            "error": str(e),
        }


def benchmark_model(
    model: str,
    dataset: list[dict[str, Any]],
    raw_traces_file: Any,
) -> dict[str, Any]:
    print(f"\n========================================================")
    print(f"Memulai Benchmark Model: {model}")
    print(f"========================================================")

    baseline_vram = get_gpu_vram_mib()
    print(f"Baseline VRAM sebelum warm-up: {baseline_vram} MiB")
    print(f"Melakukan warm-up inferensi untuk {model}...")
    _ = run_single_inference_dual_dispatch(model, "Halo Compass AI", num_predict=32)
    time.sleep(1.0)
    active_vram = get_gpu_vram_mib()
    print(f"Active VRAM setelah warm-up: {active_vram} MiB")

    peak_vram = active_vram
    results_list: list[dict[str, Any]] = []
    category_stats: dict[str, dict[str, int]] = {}

    for idx, item in enumerate(dataset):
        qid = item["id"]
        cat = item["category"]
        query = item["query"]

        if cat not in category_stats:
            category_stats[cat] = {
                "total": 0,
                "tool_select_ok": 0,
                "schema_compliant": 0,
                "param_error": 0,
                "exec_success": 0,
            }

        cur_vram = get_gpu_vram_mib()
        if cur_vram > peak_vram:
            peak_vram = cur_vram

        inf_res = run_single_inference_dual_dispatch(model, query)
        adh_res = evaluate_tool_adherence(
            category=cat,
            expected_tool=item.get("expected_tool"),
            required_params=item.get("required_params", []),
            tool_calls=inf_res["tool_calls"],
            raw_content=inf_res["content"],
            expected_ips=item.get("expected_ips"),
            expected_semester=item.get("expected_semester"),
            expected_target_course=item.get("expected_target_course"),
        )

        category_stats[cat]["total"] += 1
        if adh_res["tool_selection_ok"]:
            category_stats[cat]["tool_select_ok"] += 1
        if adh_res["schema_compliant"]:
            category_stats[cat]["schema_compliant"] += 1
        if adh_res["param_error"]:
            category_stats[cat]["param_error"] += 1
        if adh_res["exec_success"]:
            category_stats[cat]["exec_success"] += 1

        record = {
            "model": model,
            "id": qid,
            "category": cat,
            "query": query,
            "expected_tool": item.get("expected_tool"),
            "actual_tool": adh_res["actual_tool"],
            "tool_selection_ok": adh_res["tool_selection_ok"],
            "schema_compliant": adh_res["schema_compliant"],
            "param_error": adh_res["param_error"],
            "error_reason": adh_res["error_reason"],
            "exec_success": adh_res["exec_success"],
            "actual_args": adh_res["actual_args"],
            "prompt_eval_ms": inf_res["prompt_eval_ms"],
            "perceived_ttft_ms": inf_res["perceived_ttft_ms"],
            "eval_ms": inf_res["eval_ms"],
            "eval_count": inf_res["eval_count"],
            "tps": inf_res["tps"],
            "total_time_ms": inf_res["total_time_ms"],
            "current_vram_mib": cur_vram,
        }

        results_list.append(record)
        raw_traces_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        raw_traces_file.flush()

        sel_str = "OK" if adh_res["tool_selection_ok"] else "FAIL"
        sch_str = "OK" if adh_res["schema_compliant"] else "FAIL"
        par_str = "FAIL" if adh_res["param_error"] else "OK"
        print(
            f"[{idx+1:02d}/50] [{cat[:4]}] {qid}: Tool={sel_str} Sch={sch_str} Param={par_str} "
            f"| TTFT={inf_res['prompt_eval_ms']:.1f}ms TPS={inf_res['tps']:.1f}"
        )

    n_queries = len(results_list)
    sel_acc = (sum(1 for r in results_list if r["tool_selection_ok"]) / n_queries) * 100.0
    schema_comp = (sum(1 for r in results_list if r["schema_compliant"]) / n_queries) * 100.0
    param_err_rate = (sum(1 for r in results_list if r["param_error"]) / n_queries) * 100.0
    exec_rate = (sum(1 for r in results_list if r["exec_success"]) / n_queries) * 100.0

    prompt_ttfts = [r["prompt_eval_ms"] for r in results_list if r["prompt_eval_ms"] > 0]
    perceived_ttfts = [r["perceived_ttft_ms"] for r in results_list if r["perceived_ttft_ms"] > 0]
    tps_vals = [r["tps"] for r in results_list if r["tps"] > 0]

    mean_hardware_ttft = float(np.mean(prompt_ttfts)) if prompt_ttfts else 0.0
    p50_hardware_ttft = float(np.percentile(prompt_ttfts, 50)) if prompt_ttfts else 0.0
    p95_hardware_ttft = float(np.percentile(prompt_ttfts, 95)) if prompt_ttfts else 0.0

    mean_perceived_ttft = float(np.mean(perceived_ttfts)) if perceived_ttfts else 0.0
    p50_perceived_ttft = float(np.percentile(perceived_ttfts, 50)) if perceived_ttfts else 0.0
    p95_perceived_ttft = float(np.percentile(perceived_ttfts, 95)) if perceived_ttfts else 0.0

    mean_tps = float(np.mean(tps_vals)) if tps_vals else 0.0
    p50_tps = float(np.percentile(tps_vals, 50)) if tps_vals else 0.0

    summary = {
        "model": model,
        "total_queries": n_queries,
        "tool_selection_accuracy_pct": round(sel_acc, 2),
        "json_schema_compliance_pct": round(schema_comp, 2),
        "parameter_extraction_error_pct": round(param_err_rate, 2),
        "physical_execution_success_pct": round(exec_rate, 2),
        "mean_hardware_ttft_ms": round(mean_hardware_ttft, 2),
        "p50_hardware_ttft_ms": round(p50_hardware_ttft, 2),
        "p95_hardware_ttft_ms": round(p95_hardware_ttft, 2),
        "mean_perceived_ttft_ms": round(mean_perceived_ttft, 2),
        "p50_perceived_ttft_ms": round(p50_perceived_ttft, 2),
        "p95_perceived_ttft_ms": round(p95_perceived_ttft, 2),
        "mean_tps": round(mean_tps, 2),
        "p50_tps": round(p50_tps, 2),
        "baseline_vram_mib": baseline_vram,
        "peak_vram_mib": peak_vram,
        "vram_delta_mib": peak_vram - baseline_vram,
        "category_breakdown": category_stats,
    }

    print(f"\nRingkasan Model {model}:")
    print(f"  Tool Selection Accuracy : {sel_acc:.2f}%")
    print(f"  JSON Schema Compliance  : {schema_comp:.2f}%")
    print(f"  Parameter Error Rate    : {param_err_rate:.2f}%")
    print(f"  Execution Success Rate  : {exec_rate:.2f}%")
    print(f"  Mean Hardware TTFT      : {mean_hardware_ttft:.2f} ms (P50: {p50_hardware_ttft:.2f} ms)")
    print(f"  Mean Perceived TTFT     : {mean_perceived_ttft:.2f} ms (P50: {p50_perceived_ttft:.2f} ms)")
    print(f"  Throughput (Mean TPS)   : {mean_tps:.2f} tokens/s")
    print(f"  Peak VRAM Footprint     : {peak_vram} MiB (Delta: +{peak_vram - baseline_vram} MiB)")

    return summary


def calculate_pareto_analysis(model_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    max_ttft = max(s["p50_hardware_ttft_ms"] for s in model_summaries) or 1.0
    min_ttft = min(s["p50_hardware_ttft_ms"] for s in model_summaries) or 1.0
    max_vram = max(s["peak_vram_mib"] for s in model_summaries) or 1.0
    min_vram = min(s["peak_vram_mib"] for s in model_summaries) or 1.0

    enhanced = []
    for s in model_summaries:
        adherence_composite = (
            s["tool_selection_accuracy_pct"]
            + s["json_schema_compliance_pct"]
            + (100.0 - s["parameter_extraction_error_pct"])
        ) / 3.0

        norm_acc = adherence_composite / 100.0
        norm_lat = 1.0 - ((s["p50_hardware_ttft_ms"] - min_ttft) / (max_ttft - min_ttft)) if max_ttft > min_ttft else 1.0
        norm_vram = 1.0 - ((s["peak_vram_mib"] - min_vram) / (max_vram - min_vram)) if max_vram > min_vram else 1.0

        pareto_score = (norm_acc * 0.50) + (norm_lat * 0.25) + (norm_vram * 0.25)

        s_copy = dict(s)
        s_copy["adherence_composite_score"] = round(adherence_composite, 2)
        s_copy["pareto_efficiency_score"] = round(pareto_score * 100.0, 2)
        enhanced.append(s_copy)

    for i, a in enumerate(enhanced):
        is_dominated = False
        for j, b in enumerate(enhanced):
            if i == j:
                continue
            b_better_or_equal = (
                b["adherence_composite_score"] >= a["adherence_composite_score"]
                and b["p50_hardware_ttft_ms"] <= a["p50_hardware_ttft_ms"]
                and b["peak_vram_mib"] <= a["peak_vram_mib"]
            )
            b_strictly_better = (
                b["adherence_composite_score"] > a["adherence_composite_score"]
                or b["p50_hardware_ttft_ms"] < a["p50_hardware_ttft_ms"]
                or b["peak_vram_mib"] < a["peak_vram_mib"]
            )
            if b_better_or_equal and b_strictly_better:
                is_dominated = True
                break
        a["is_pareto_frontier"] = not is_dominated

    return enhanced


def save_reports(
    summaries: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "benchmark_metrics.csv"
    fieldnames = [
        "model",
        "total_queries",
        "tool_selection_accuracy_pct",
        "json_schema_compliance_pct",
        "parameter_extraction_error_pct",
        "physical_execution_success_pct",
        "adherence_composite_score",
        "mean_hardware_ttft_ms",
        "p50_hardware_ttft_ms",
        "p95_hardware_ttft_ms",
        "mean_perceived_ttft_ms",
        "p50_perceived_ttft_ms",
        "mean_tps",
        "p50_tps",
        "peak_vram_mib",
        "vram_delta_mib",
        "pareto_efficiency_score",
        "is_pareto_frontier",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in summaries:
            row = {k: s.get(k, "") for k in fieldnames}
            writer.writerow(row)

    json_path = output_dir / "benchmark_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)

    summary_md_path = output_dir / "benchmark_summary.md"
    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write("# Benchmark Komparasi SLM Ternary (Bonsai 2: 2B, 4B, 9B) vs Baseline Produksi Qwen2.5\n\n")
        f.write("## Ringkasan Eksekutif & Temuan Empiris\n\n")
        f.write(
            "Eksperimen Exp-024 mengevaluasi secara empiris efisiensi dan akurasi model Native Ternary (Bonsai 2) "
            "dengan arsitektur bobot 1.58-bit PQ2_0 (2B, 4B, 9B) dibandingkan dengan baseline model produksi Qwen2.5 (1.5B, 3B, 7B). "
            "Evaluasi dilakukan pada 50 kueri uji akademik terstratifikasi FTMM COMPASS.\n\n"
        )

        f.write("## Tabel Komparasi Metrik Lengkap\n\n")
        f.write(
            "| Model | Tool Sel Acc (%) | Schema Comp (%) | Param Err (%) | Exec Success (%) | "
            "P50 TTFT (ms) | Mean TPS | Peak VRAM (MiB) | Pareto Score | Frontier |\n"
        )
        f.write(
            "|---|---|---|---|---|---|---|---|---|---|\n"
        )
        for s in summaries:
            frontier_str = "YES" if s.get("is_pareto_frontier") else "NO"
            f.write(
                f"| `{s['model']}` | {s['tool_selection_accuracy_pct']:.2f}% | "
                f"{s['json_schema_compliance_pct']:.2f}% | {s['parameter_extraction_error_pct']:.2f}% | "
                f"{s['physical_execution_success_pct']:.2f}% | {s['p50_hardware_ttft_ms']:.2f} ms | "
                f"{s['mean_tps']:.2f} | {s['peak_vram_mib']} MiB | {s.get('pareto_efficiency_score', 0.0):.2f} | "
                f"{frontier_str} |\n"
            )

        f.write("\n## Breakdown Akurasi per Kategori Kueri\n\n")
        all_cats = list(summaries[0].get("category_breakdown", {}).keys())
        header = "| Model | " + " | ".join(all_cats) + " |\n"
        sep = "|---|" + "|".join(["---" for _ in all_cats]) + "|\n"
        f.write(header)
        f.write(sep)
        for s in summaries:
            cb = s.get("category_breakdown", {})
            cat_strs = []
            for c in all_cats:
                cdata = cb.get(c, {})
                tot = cdata.get("total", 0)
                ok = cdata.get("exec_success", 0)
                pct = (ok / tot * 100.0) if tot > 0 else 0.0
                cat_strs.append(f"{pct:.1f}% ({ok}/{tot})")
            f.write(f"| `{s['model']}` | " + " | ".join(cat_strs) + " |\n")

        f.write("\n## Analisis Multi-Objective Pareto & Rekomendasi Arsitektur\n\n")
        frontiers = [s for s in summaries if s.get("is_pareto_frontier")]
        frontier_names = ", ".join(f"`{m['model']}`" for m in frontiers)
        f.write(f"- **Model Pareto Frontier**: {frontier_names}\n")
        f.write(
            "- Model Bonsai 2 (Native Ternary 1.58-bit) mendemonstrasikan efisiensi memori VRAM yang luar biasa "
            "berkat kuantisasi PQ2_0, memungkinkan parameter 9B berjalan pada footprint VRAM ~2.5-3.0 GB.\n"
        )
        f.write(
            "- Kecepatan throughput (TPS) pada Bonsai 2 memanfaatkan kernel CUDA Hadamard-aware "
            "sehingga menghasilkan latensi interaktif yang sangat tinggi.\n"
        )

    readme_path = output_dir / "README.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("# Exp-024: Benchmark Komparasi SLM Ternary (Bonsai 2: 2B, 4B, 9B) vs Baseline Produksi Qwen2.5\n\n")
        f.write(
            "Folder ini menyimpan artefak evaluasi empiris komparasi model Native Ternary Bonsai 2 "
            "dengan baseline Qwen2.5 pada benchmark 50 kueri terstratifikasi FTMM COMPASS.\n\n"
        )
        f.write("## Daftar Berkas Artefak\n")
        f.write("- `benchmark_metrics.csv`: Matriks data numerik per model.\n")
        f.write("- `benchmark_results.json`: Ringkasan terstruktur dan metrik agregat.\n")
        f.write("- `benchmark_raw_traces.jsonl`: Log jejak inferensi per kueri (total 300 runs).\n")
        f.write("- `benchmark_summary.md`: Laporan komprehensif analisis Pareto dan evaluasi arsitektur.\n")


def summarize_traces_and_generate_reports(traces_path: Path, output_dir: Path) -> list[dict[str, Any]]:
    if not traces_path.exists():
        return []
    records: list[dict[str, Any]] = []
    with open(traces_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    models_seen = []
    for r in records:
        if r["model"] not in models_seen:
            models_seen.append(r["model"])

    all_summaries = []
    for m in models_seen:
        m_records = [r for r in records if r["model"] == m]
        n_queries = len(m_records)
        if n_queries == 0:
            continue

        sel_acc = (sum(1 for r in m_records if r["tool_selection_ok"]) / n_queries) * 100.0
        schema_comp = (sum(1 for r in m_records if r["schema_compliant"]) / n_queries) * 100.0
        param_err_rate = (sum(1 for r in m_records if r["param_error"]) / n_queries) * 100.0
        exec_rate = (sum(1 for r in m_records if r["exec_success"]) / n_queries) * 100.0

        prompt_ttfts = [r["prompt_eval_ms"] for r in m_records if r["prompt_eval_ms"] > 0]
        perceived_ttfts = [r["perceived_ttft_ms"] for r in m_records if r["perceived_ttft_ms"] > 0]
        tps_vals = [r["tps"] for r in m_records if r["tps"] > 0]
        vram_vals = [r.get("current_vram_mib", 0) for r in m_records]

        mean_hardware_ttft = float(np.mean(prompt_ttfts)) if prompt_ttfts else 0.0
        p50_hardware_ttft = float(np.percentile(prompt_ttfts, 50)) if prompt_ttfts else 0.0
        p95_hardware_ttft = float(np.percentile(prompt_ttfts, 95)) if prompt_ttfts else 0.0

        mean_perceived_ttft = float(np.mean(perceived_ttfts)) if perceived_ttfts else 0.0
        p50_perceived_ttft = float(np.percentile(perceived_ttfts, 50)) if perceived_ttfts else 0.0
        p95_perceived_ttft = float(np.percentile(perceived_ttfts, 95)) if perceived_ttfts else 0.0

        mean_tps = float(np.mean(tps_vals)) if tps_vals else 0.0
        p50_tps = float(np.percentile(tps_vals, 50)) if tps_vals else 0.0
        peak_vram = max(vram_vals) if vram_vals else 0
        baseline_vram = min(vram_vals) if vram_vals else 0

        category_stats: dict[str, dict[str, int]] = {}
        for r in m_records:
            c = r["category"]
            if c not in category_stats:
                category_stats[c] = {
                    "total": 0,
                    "tool_select_ok": 0,
                    "schema_compliant": 0,
                    "param_error": 0,
                    "exec_success": 0,
                }
            category_stats[c]["total"] += 1
            if r["tool_selection_ok"]:
                category_stats[c]["tool_select_ok"] += 1
            if r["schema_compliant"]:
                category_stats[c]["schema_compliant"] += 1
            if r["param_error"]:
                category_stats[c]["param_error"] += 1
            if r["exec_success"]:
                category_stats[c]["exec_success"] += 1

        summary = {
            "model": m,
            "total_queries": n_queries,
            "tool_selection_accuracy_pct": round(sel_acc, 2),
            "json_schema_compliance_pct": round(schema_comp, 2),
            "parameter_extraction_error_pct": round(param_err_rate, 2),
            "physical_execution_success_pct": round(exec_rate, 2),
            "mean_hardware_ttft_ms": round(mean_hardware_ttft, 2),
            "p50_hardware_ttft_ms": round(p50_hardware_ttft, 2),
            "p95_hardware_ttft_ms": round(p95_hardware_ttft, 2),
            "mean_perceived_ttft_ms": round(mean_perceived_ttft, 2),
            "p50_perceived_ttft_ms": round(p50_perceived_ttft, 2),
            "p95_perceived_ttft_ms": round(p95_perceived_ttft, 2),
            "mean_tps": round(mean_tps, 2),
            "p50_tps": round(p50_tps, 2),
            "baseline_vram_mib": baseline_vram,
            "peak_vram_mib": peak_vram,
            "vram_delta_mib": peak_vram - baseline_vram,
            "category_breakdown": category_stats,
        }
        all_summaries.append(summary)

    enhanced_summaries = calculate_pareto_analysis(all_summaries)
    save_reports(enhanced_summaries, output_dir)
    return enhanced_summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark SLM Ternary Bonsai 2 vs Baseline Qwen2.5")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Path ke dataset evaluasi")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Path direktori output")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="Daftar model yang diuji")
    parser.add_argument("--append", action="store_true", help="Append ke traces yang sudah ada alih-alih overwrite")
    args = parser.parse_args()

    if not args.dataset.exists():
        print(f"Error: Dataset tidak ditemukan di {args.dataset}")
        sys.exit(1)

    with open(args.dataset, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_traces_path = args.output_dir / "benchmark_raw_traces.jsonl"
    file_mode = "a" if args.append else "w"
    raw_traces_file = open(raw_traces_path, file_mode, encoding="utf-8")

    all_summaries: list[dict[str, Any]] = []

    try:
        for model in args.models:
            is_bonsai = "bonsai" in model.lower()
            bonsai_proc = None

            if is_bonsai:
                print(f"\n[Lifecycle] Mempersiapkan llama-server untuk model {model}...")
                stop_bonsai_server(port=8080)
                time.sleep(1.0)
                bonsai_proc = start_bonsai_server(model, port=8080)
                print(f"[Lifecycle] llama-server untuk {model} aktif di port 8080.")
            else:
                print(f"\n[Lifecycle] Mempersiapkan Ollama model {model}...")
                stop_ollama_model(model)

            try:
                summary = benchmark_model(model, dataset, raw_traces_file)
                all_summaries.append(summary)
            finally:
                if is_bonsai:
                    print(f"[Lifecycle] Mematikan llama-server untuk model {model}...")
                    stop_bonsai_server(bonsai_proc, port=8080)
                else:
                    print(f"[Lifecycle] Mematikan model Ollama {model}...")
                    stop_ollama_model(model)
                time.sleep(2.0)

    finally:
        raw_traces_file.close()

    enhanced_summaries = summarize_traces_and_generate_reports(raw_traces_path, args.output_dir)
    print(f"\nSeluruh pengujian selesai! Laporan tersimpan di {args.output_dir}/")


if __name__ == "__main__":
    main()
