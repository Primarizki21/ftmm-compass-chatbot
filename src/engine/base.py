"""
Definisi tipe data dan kontrak dasar engine generasi FTMM COMPASS.
"""

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator
import httpx
import ollama

logger = logging.getLogger(__name__)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
DEFAULT_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "2048"))
LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama").strip().lower()
BONSAI_HOST = os.getenv("BONSAI_HOST", "http://127.0.0.1:8080").rstrip("/")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")

_OLLAMA_CLIENT_POOL: dict[str, ollama.Client] = {}


def get_ollama_client() -> ollama.Client:
    host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").strip()
    if not host.startswith(("http://", "https://")):
        host = f"http://{host}"
    normalized_host = host.rstrip("/")
    client = _OLLAMA_CLIENT_POOL.get(normalized_host)
    if client is None:
        client = ollama.Client(host=normalized_host)
        _OLLAMA_CLIENT_POOL[normalized_host] = client
    return client


class NormalizedToolCallFunction:
    def __init__(self, name: str, arguments: dict[str, Any] | str):
        self.name = name
        if isinstance(arguments, str):
            try:
                self.arguments = json.loads(arguments)
            except Exception:
                self.arguments = {}
        elif isinstance(arguments, dict):
            self.arguments = arguments
        else:
            self.arguments = {}


class NormalizedToolCall:
    def __init__(
        self,
        name: str,
        arguments: dict[str, Any] | str,
        call_id: str | None = None,
        tool_type: str = "function",
    ):
        self.id = call_id or f"call_{int(time.time() * 1000)}"
        self.type = tool_type
        self.function = NormalizedToolCallFunction(name=name, arguments=arguments)


class NormalizedChatMessage:
    def __init__(
        self,
        role: str = "assistant",
        content: str = "",
        tool_calls: list[NormalizedToolCall] | None = None,
    ):
        self.role = role
        self.content = content or ""
        self.tool_calls = tool_calls


class NormalizedChatResponse:
    def __init__(
        self,
        message: NormalizedChatMessage,
        prompt_eval_count: int = 0,
        eval_count: int = 0,
        model: str = "",
        backend: str = "bonsai",
        raw_response: dict[str, Any] | None = None,
    ):
        self.message = message
        self.prompt_eval_count = prompt_eval_count
        self.eval_count = eval_count
        self.model = model
        self.backend = backend
        self.raw_response = raw_response or {}

    def get(self, key: str, default: Any = None) -> Any:
        if key == "prompt_eval_count":
            return self.prompt_eval_count
        if key == "eval_count":
            return self.eval_count
        if key == "message":
            return self.message
        if key == "content":
            return self.message.content
        if key == "tool_calls":
            return self.message.tool_calls
        if key == "model":
            return self.model
        if key == "backend":
            return self.backend
        if isinstance(self.raw_response, dict):
            return self.raw_response.get(key, default)
        return default

    def __getitem__(self, key: str) -> Any:
        val = self.get(key)
        if val is None:
            if isinstance(self.raw_response, dict) and key in self.raw_response:
                return self.raw_response[key]
            raise KeyError(key)
        return val


class LLMClientAdapter:
    """Adapter multi-backend LLM: Bonsai-2-4B OpenAI-compatible dengan graceful fallback ke Ollama."""

    def __init__(
        self,
        backend: str | None = None,
        bonsai_host: str | None = None,
        ollama_host: str | None = None,
        fallback_model: str = "llama3.2:3b",
        timeout: float = 60.0,
        enable_fallback: bool | None = None,
    ):
        self.backend = (backend or os.getenv("LLM_BACKEND", "ollama")).strip().lower()
        b_host = bonsai_host or os.getenv("BONSAI_HOST", "http://127.0.0.1:8080")
        if not b_host.startswith(("http://", "https://")):
            b_host = f"http://{b_host}"
        self.bonsai_host = b_host.rstrip("/")

        o_host = ollama_host or os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
        if not o_host.startswith(("http://", "https://")):
            o_host = f"http://{o_host}"
        self.ollama_host = o_host.rstrip("/")

        self.fallback_model = fallback_model
        self.timeout = timeout
        if enable_fallback is not None:
            self.enable_fallback = enable_fallback
        else:
            self.enable_fallback = os.getenv("ENABLE_OLLAMA_FALLBACK", "false").strip().lower() in ("true", "1")
    @staticmethod
    def _normalize_messages(messages: list[Any], for_openai: bool = True) -> list[dict[str, Any]]:
        clean: list[dict[str, Any]] = []
        for m in messages:
            if isinstance(m, NormalizedChatMessage) or hasattr(m, "role"):
                item: dict[str, Any] = {"role": m.role, "content": getattr(m, "content", "")}
                tc_list = getattr(m, "tool_calls", None)
                if tc_list:
                    formatted_tcs = []
                    for idx, tc in enumerate(tc_list):
                        fn_name = tc.function.name
                        fn_args = tc.function.arguments
                        if for_openai:
                            formatted_args = (
                                json.dumps(fn_args, ensure_ascii=False)
                                if isinstance(fn_args, dict)
                                else str(fn_args)
                            )
                        else:
                            if isinstance(fn_args, str):
                                try:
                                    formatted_args = json.loads(fn_args)
                                except Exception:
                                    formatted_args = {}
                            elif isinstance(fn_args, dict):
                                formatted_args = fn_args
                            else:
                                formatted_args = {}
                        entry: dict[str, Any] = {
                            "type": "function",
                            "function": {
                                "name": fn_name,
                                "arguments": formatted_args,
                            },
                        }
                        if for_openai:
                            entry["id"] = getattr(tc, "id", f"call_{idx}") or f"call_{idx}"
                        formatted_tcs.append(entry)
                    item["tool_calls"] = formatted_tcs
                clean.append(item)
            elif isinstance(m, dict):
                d = dict(m)
                if for_openai and d.get("role") == "tool" and "tool_call_id" not in d:
                    d["tool_call_id"] = "call_0"
                if "tool_calls" in d and d["tool_calls"]:
                    formatted_tcs = []
                    for idx, tc in enumerate(d["tool_calls"]):
                        fn = tc.get("function", {}) if isinstance(tc, dict) else getattr(tc, "function", {})
                        fn_name = fn.get("name") if isinstance(fn, dict) else getattr(fn, "name", "")
                        fn_args = fn.get("arguments", {}) if isinstance(fn, dict) else getattr(fn, "arguments", {})
                        if for_openai:
                            formatted_args = (
                                json.dumps(fn_args, ensure_ascii=False)
                                if isinstance(fn_args, dict)
                                else str(fn_args)
                            )
                        else:
                            if isinstance(fn_args, str):
                                try:
                                    formatted_args = json.loads(fn_args)
                                except Exception:
                                    formatted_args = {}
                            elif isinstance(fn_args, dict):
                                formatted_args = fn_args
                            else:
                                formatted_args = {}
                        entry = {
                            "type": "function",
                            "function": {"name": fn_name, "arguments": formatted_args},
                        }
                        if for_openai:
                            entry["id"] = tc.get("id", f"call_{idx}") if isinstance(tc, dict) else f"call_{idx}"
                        formatted_tcs.append(entry)
                    d["tool_calls"] = formatted_tcs
                clean.append(d)

        # Consolidate multiple system messages into a single leading system message
        # to prevent strict Jinja template errors (e.g. llama-server loop.first invariant).
        system_chunks: list[str] = []
        non_system_messages: list[dict[str, Any]] = []
        for item in clean:
            if item.get("role") == "system":
                c = str(item.get("content", "")).strip()
                if c:
                    system_chunks.append(c)
            else:
                non_system_messages.append(item)
        if system_chunks:
            return [{"role": "system", "content": "\n\n".join(system_chunks)}] + non_system_messages
        return non_system_messages

    @staticmethod
    def _parse_tool_calls(
        content: str,
        raw_tool_calls: list[dict[str, Any]] | None = None,
        reasoning: str = "",
    ) -> list[NormalizedToolCall] | None:
        # 1. Direct tool_calls dari API
        if raw_tool_calls:
            parsed: list[NormalizedToolCall] = []
            for tc in raw_tool_calls:
                fn = tc.get("function", {}) if isinstance(tc, dict) else getattr(tc, "function", {})
                fn_name = fn.get("name") if isinstance(fn, dict) else getattr(fn, "name", None)
                args_raw = fn.get("arguments", {}) if isinstance(fn, dict) else getattr(fn, "arguments", {})
                call_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
                if fn_name:
                    parsed.append(NormalizedToolCall(name=fn_name, arguments=args_raw, call_id=call_id))
            if parsed:
                return parsed

        # 2. Parsing teks / markdown / JSON / tags
        search_texts = [content]
        if reasoning:
            search_texts.append(reasoning)

        registered_tools = {
            "search_courses",
            "validate_prerequisites_dag",
            "calculate_sks_limit",
            "generate_degree_plan",
            "get_courses_by_semester",
            "search_academic_regulations",
        }

        for text in search_texts:
            if not text:
                continue

            # Markdown JSON blocks
            json_blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
            for block in json_blocks:
                block = block.strip()
                try:
                    data = json.loads(block)
                    if isinstance(data, dict):
                        if "action" in data and str(data["action"]).strip() in registered_tools:
                            return [NormalizedToolCall(name=str(data["action"]).strip(), arguments=data.get("action_input", {}))]
                        if "name" in data and str(data["name"]).strip() in registered_tools:
                            return [NormalizedToolCall(name=str(data["name"]).strip(), arguments=data.get("arguments", {}))]
                        if "tool" in data and str(data["tool"]).strip() in registered_tools:
                            return [NormalizedToolCall(name=str(data["tool"]).strip(), arguments=data.get("parameters", data.get("args", {})))]
                        for r_tool in registered_tools:
                            if r_tool in data and isinstance(data[r_tool], dict):
                                return [NormalizedToolCall(name=r_tool, arguments=data[r_tool])]
                    elif isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict) and "name" in item and str(item["name"]).strip() in registered_tools:
                                return [NormalizedToolCall(name=str(item["name"]).strip(), arguments=item.get("arguments", {}))]
                except Exception:
                    continue

            # Tool call tags <tool_call>...</tool_call>
            tool_tags = re.findall(r"<tool_call>\s*([\s\S]*?)\s*</tool_call>", text, re.IGNORECASE)
            for tag_content in tool_tags:
                try:
                    data = json.loads(tag_content.strip())
                    if isinstance(data, dict):
                        name = data.get("name") or data.get("action") or data.get("tool")
                        args = data.get("arguments") or data.get("action_input") or data.get("parameters", {})
                        if name and str(name).strip() in registered_tools:
                            return [NormalizedToolCall(name=str(name).strip(), arguments=args)]
                except Exception:
                    continue

            # Qwen / Bonsai function calling pseudo-tags: <function=name>...<parameter=key>value</parameter>...</function>
            fn_matches = re.findall(
                r"<function=([a-zA-Z0-9_]+)>\s*([\s\S]*?)\s*</function>",
                text,
                re.IGNORECASE,
            )
            for fn_name, params_raw in fn_matches:
                fn_name = fn_name.strip()
                if fn_name in registered_tools:
                    param_pairs = re.findall(
                        r"<parameter=([a-zA-Z0-9_]+)>\s*([\s\S]*?)\s*</parameter>",
                        params_raw,
                        re.IGNORECASE,
                    )
                    args_dict: dict[str, Any] = {}
                    for p_key, p_val in param_pairs:
                        p_val_clean = p_val.strip()
                        if p_val_clean.isdigit():
                            args_dict[p_key] = int(p_val_clean)
                        else:
                            args_dict[p_key] = p_val_clean
                    return [NormalizedToolCall(name=fn_name, arguments=args_dict)]

            # ReAct pattern: Action: <name>\nAction Input: <json>
            react_match = re.search(
                r"Action:\s*([a-zA-Z0-9_]+)\s*\n\s*Action\s*Input:\s*([\s\S]*?)(?=\n\s*(?:Observation|Action|Final Answer)|$)",
                text,
                re.IGNORECASE,
            )
            if react_match:
                tool_name = react_match.group(1).strip()
                if tool_name in registered_tools:
                    raw_input = react_match.group(2).strip()
                    try:
                        args = json.loads(raw_input)
                    except Exception:
                        args = {}
                    return [NormalizedToolCall(name=tool_name, arguments=args)]

            # Raw JSON regex
            raw_json_match = re.search(r"(\{\s*\"(?:action|name|tool)\"\s*:\s*\"[a-zA-Z0-9_]+\"[\s\S]*?\})", text)
            if raw_json_match:
                try:
                    data = json.loads(raw_json_match.group(1))
                    name = data.get("action") or data.get("name") or data.get("tool")
                    args = data.get("action_input") or data.get("arguments") or data.get("parameters", {})
                    if name and str(name).strip() in registered_tools:
                        return [NormalizedToolCall(name=str(name).strip(), arguments=args)]
                except Exception:
                    pass

        return None

    def chat(
        self,
        messages: list[Any],
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        options: dict[str, Any] | None = None,
        keep_alive: Any = None,
        **kwargs: Any,
    ) -> NormalizedChatResponse:
        if self.backend == "bonsai":
            try:
                return self._chat_bonsai(
                    messages=messages,
                    model=model,
                    tools=tools,
                    options=options,
                    **kwargs,
                )
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.TimeoutException, httpx.HTTPStatusError, Exception) as e:
                if not self.enable_fallback:
                    logger.error(
                        "Bonsai server (%s) request failed: %s (Ollama failover disabled to prevent WSL2 multi-generator OOM)",
                        self.bonsai_host,
                        e,
                    )
                    raise e
                logger.warning(
                    "Bonsai server (%s) unreachable or failed: %s. Gracefully falling back to Ollama (%s) with model %s",
                    self.bonsai_host,
                    e,
                    self.ollama_host,
                    self.fallback_model,
                )
                return self._chat_ollama(
                    messages=messages,
                    model=self.fallback_model,
                    tools=tools,
                    options=options,
                    keep_alive=keep_alive,
                    **kwargs,
                )

        return self._chat_ollama(
            messages=messages,
            model=model,
            tools=tools,
            options=options,
            keep_alive=keep_alive,
            **kwargs,
        )

    def _chat_bonsai(
        self,
        messages: list[Any],
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        options: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> NormalizedChatResponse:
        target_model = model or os.getenv("LLM_MODEL_NAME", "bonsai-2-4b")
        clean_messages = self._normalize_messages(messages, for_openai=True)

        num_predict = 1024
        temperature = 0.1
        top_p = 0.9
        if options:
            num_predict = options.get("num_predict", 1024)
            temperature = options.get("temperature", 0.1)
            top_p = options.get("top_p", 0.9)

        payload: dict[str, Any] = {
            "model": target_model,
            "messages": clean_messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": num_predict,
        }
        if tools:
            payload["tools"] = tools

        url = f"{self.bonsai_host}/v1/chat/completions"
        with httpx.Client(timeout=httpx.Timeout(self.timeout, connect=3.0)) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()

        choice = data.get("choices", [{}])[0]
        msg_dict = choice.get("message", {})
        content = msg_dict.get("content", "") or ""
        raw_tool_calls = msg_dict.get("tool_calls")
        reasoning = msg_dict.get("reasoning_content", "") or ""
        if not content and reasoning and not raw_tool_calls:
            ans_marker = re.search(r"(?:Draft|Jawaban|Final Answer|Summary):\s*([\s\S]+)$", reasoning, re.IGNORECASE)
            if ans_marker:
                content = ans_marker.group(1).strip()
            else:
                content = reasoning.strip()
        parsed_tool_calls = self._parse_tool_calls(
            content=content,
            raw_tool_calls=raw_tool_calls,
            reasoning=reasoning,
        )

        usage = data.get("usage", {})
        prompt_eval_count = usage.get("prompt_tokens", 0)
        eval_count = usage.get("completion_tokens", 0)

        chat_msg = NormalizedChatMessage(
            role="assistant",
            content=content,
            tool_calls=parsed_tool_calls,
        )

        return NormalizedChatResponse(
            message=chat_msg,
            prompt_eval_count=prompt_eval_count,
            eval_count=eval_count,
            model=target_model,
            backend="bonsai",
            raw_response=data,
        )

    def _chat_ollama(
        self,
        messages: list[Any],
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        options: dict[str, Any] | None = None,
        keep_alive: Any = None,
        **kwargs: Any,
    ) -> NormalizedChatResponse:
        target_model = model or os.getenv("LLM_MODEL_NAME", "llama3.2:3b")
        clean_messages = self._normalize_messages(messages, for_openai=False)
        client = get_ollama_client()

        chat_kwargs: dict[str, Any] = {
            "model": target_model,
            "messages": clean_messages,
        }
        if tools:
            chat_kwargs["tools"] = tools
        if options:
            chat_kwargs["options"] = options
        if keep_alive is not None:
            chat_kwargs["keep_alive"] = keep_alive

        resp = client.chat(**chat_kwargs)

        prompt_eval_count = (
            resp.get("prompt_eval_count", 0)
            if hasattr(resp, "get")
            else getattr(resp, "prompt_eval_count", 0)
        )
        eval_count = (
            resp.get("eval_count", 0)
            if hasattr(resp, "get")
            else getattr(resp, "eval_count", 0)
        )

        msg = resp.message if hasattr(resp, "message") else (resp.get("message") if hasattr(resp, "get") else None)
        content = getattr(msg, "content", "") if msg else ""
        raw_tool_calls = getattr(msg, "tool_calls", None) if msg else None

        parsed_tool_calls: list[NormalizedToolCall] | None = None
        if raw_tool_calls:
            parsed_tool_calls = []
            for tc in raw_tool_calls:
                fn = getattr(tc, "function", None) or (tc.get("function") if isinstance(tc, dict) else {})
                fn_name = getattr(fn, "name", None) or (fn.get("name") if isinstance(fn, dict) else None)
                args_raw = getattr(fn, "arguments", {}) or (fn.get("arguments") if isinstance(fn, dict) else {})
                if fn_name:
                    parsed_tool_calls.append(NormalizedToolCall(name=fn_name, arguments=args_raw))
        else:
            parsed_tool_calls = self._parse_tool_calls(content=content)

        chat_msg = NormalizedChatMessage(
            role="assistant",
            content=content,
            tool_calls=parsed_tool_calls,
        )

        return NormalizedChatResponse(
            message=chat_msg,
            prompt_eval_count=prompt_eval_count,
            eval_count=eval_count,
            model=target_model,
            backend="ollama",
            raw_response=resp if isinstance(resp, dict) else getattr(resp, "__dict__", {}),
        )

    async def stream_chat(
        self,
        messages: list[Any],
        model: str | None = None,
        options: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        if self.backend == "bonsai":
            try:
                async for chunk in self._stream_chat_bonsai(messages=messages, model=model, options=options, **kwargs):
                    yield chunk
                return
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.TimeoutException, httpx.HTTPStatusError, Exception) as e:
                if not self.enable_fallback:
                    logger.error("Bonsai server stream request failed: %s (Ollama failover disabled)", e)
                    raise e
                logger.warning("Bonsai stream failed: %s. Falling back to Ollama.", e)
                async for chunk in self._stream_chat_ollama(messages=messages, model=self.fallback_model, options=options, **kwargs):
                    yield chunk
                return

        async for chunk in self._stream_chat_ollama(messages=messages, model=model, options=options, **kwargs):
            yield chunk

    async def _stream_chat_bonsai(
        self,
        messages: list[Any],
        model: str | None = None,
        options: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        target_model = model or os.getenv("LLM_MODEL_NAME", "bonsai-2-4b")
        clean_messages = self._normalize_messages(messages, for_openai=True)
        num_predict = 1024
        temperature = 0.1
        top_p = 0.9
        if options:
            num_predict = options.get("num_predict", 1024)
            temperature = options.get("temperature", 0.1)
            top_p = options.get("top_p", 0.9)

        payload = {
            "model": target_model,
            "messages": clean_messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": num_predict,
            "stream": True,
        }
        url = f"{self.bonsai_host}/v1/chat/completions"
        thought_filter = StreamingThoughtFilter()

        async with httpx.AsyncClient(timeout=httpx.Timeout(self.timeout, connect=5.0)) as client:
            async with client.stream("POST", url, json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    raw_data = line[5:].strip()
                    if raw_data == "[DONE]":
                        break
                    try:
                        chunk_json = json.loads(raw_data)
                        delta = chunk_json.get("choices", [{}])[0].get("delta", {})
                        token = delta.get("content") or ""
                        if token:
                            filtered = thought_filter.process(token)
                            if filtered:
                                yield filtered
                    except Exception:
                        continue
        flushed = thought_filter.flush()
        if flushed:
            yield flushed

    async def _stream_chat_ollama(
        self,
        messages: list[Any],
        model: str | None = None,
        options: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        target_model = model or os.getenv("LLM_MODEL_NAME", "llama3.2:3b")
        clean_messages = self._normalize_messages(messages, for_openai=False)
        payload = {
            "model": target_model,
            "messages": clean_messages,
            "stream": True,
        }
        if options:
            payload["options"] = options
        url = f"{self.ollama_host}/api/chat"
        thought_filter = StreamingThoughtFilter()

        async with httpx.AsyncClient(timeout=httpx.Timeout(self.timeout, connect=5.0)) as client:
            async with client.stream("POST", url, json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    try:
                        chunk_json = json.loads(line)
                        token = chunk_json.get("message", {}).get("content", "")
                        if token:
                            filtered = thought_filter.process(token)
                            if filtered:
                                yield filtered
                        if chunk_json.get("done"):
                            break
                    except Exception:
                        continue
        flushed = thought_filter.flush()
        if flushed:
            yield flushed


class StreamingThoughtFilter:
    """Filter real-time token stream untuk mengeliminasi tag internal <think>...</think>."""

    def __init__(self) -> None:
        self.in_think = False
        self.buffer = ""
        self.header_checked = False

    def process(self, token: str) -> str:
        if not self.header_checked:
            self.buffer += token
            if "<think>" in self.buffer:
                self.in_think = True
                self.buffer = self.buffer.split("<think>", 1)[1]
                self.header_checked = True
            elif len(self.buffer) >= 8:
                self.header_checked = True
                out = self.buffer
                self.buffer = ""
                return out
            else:
                return ""

        if self.in_think:
            self.buffer += token
            if "</think>" in self.buffer:
                self.in_think = False
                out = self.buffer.split("</think>", 1)[1].lstrip("\n ")
                self.buffer = ""
                return out
            return ""

        return token

    def flush(self) -> str:
        if self.in_think:
            return ""
        out = self.buffer
        self.buffer = ""
        return out


_LLM_CLIENT_INSTANCE: LLMClientAdapter | None = None


def get_llm_client(backend: str | None = None, force_new: bool = False) -> LLMClientAdapter:
    global _LLM_CLIENT_INSTANCE
    effective_backend = (backend or os.getenv("LLM_BACKEND", "ollama")).strip().lower()
    if _LLM_CLIENT_INSTANCE is None or force_new or (_LLM_CLIENT_INSTANCE.backend != effective_backend):
        _LLM_CLIENT_INSTANCE = LLMClientAdapter(backend=effective_backend)
    return _LLM_CLIENT_INSTANCE

def clean_thinking_content(text: str) -> str:
    """Ekstraksi dan pembersihan tag <think>...</think>, unclosed <think>, serta blok Thinking Process."""
    if not text:
        return ""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    cleaned = re.sub(r"<think>.*$", "", cleaned, flags=re.DOTALL).strip()
    if not cleaned and "<think>" in text:
        cleaned = re.sub(r"</?think>", "", text, flags=re.IGNORECASE).strip()
    if cleaned.startswith("Thinking Process:") or cleaned.startswith("**Thinking Process:**"):
        ans_match = re.search(
            r"(?:Berikut|Berdasarkan|Panduan|Syarat|Kesimpulan|Jawaban|Final Answer)[\s\S]+$",
            cleaned,
            re.IGNORECASE,
        )
        if ans_match:
            cleaned = ans_match.group(0).strip()
        else:
            cleaned = ""
    return cleaned

@dataclass
class EngineResult:
    answer: str
    architecture: str                     # "Pure RAG" | "Pure CAG" | "Hybrid RAG-CAG"
    retrieved_chunks: list[str] = field(default_factory=list)
    retrieval_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    tokens_generated: int = 0
    routing_mode: str = "DEFAULT"
    metadata: dict[str, Any] = field(default_factory=dict)
