import logging
import os
from typing import Any
import ollama

from src.engine.base import get_ollama_client

logger = logging.getLogger(__name__)

DEFAULT_HYDE_MODEL = os.getenv("HYDE_MODEL_NAME") or os.getenv("LLM_MODEL_NAME", "llama3.2:3b")
DEFAULT_MAX_TOKENS = int(os.getenv("HYDE_MAX_TOKENS", "100"))
DEFAULT_TEMPERATURE = float(os.getenv("HYDE_TEMPERATURE", "0.3"))


class HydeGenerator:
    def __init__(
        self,
        model_name: str | None = None,
        host: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ):
        self.model_name = model_name or DEFAULT_HYDE_MODEL
        self.max_tokens = max_tokens if max_tokens is not None else DEFAULT_MAX_TOKENS
        self.temperature = temperature if temperature is not None else DEFAULT_TEMPERATURE
        self.host = host
        self._client: ollama.Client | None = None

    @property
    def client(self) -> ollama.Client:
        if self._client is None:
            if self.host:
                self._client = ollama.Client(host=self.host)
            else:
                self._client = get_ollama_client()
        return self._client

    def build_prompt(self, query: str, prodi_hint: str | None = None) -> str:
        prodi_ctx = f" di Fakultas Teknologi Maju dan Multidisiplin (FTMM) UNAIR, khususnya program studi {prodi_hint}," if prodi_hint else " di lingkungan akademik FTMM UNAIR,"
        return (
            f"Anda adalah sistem kurikulum akademik FTMM UNAIR. Tuliskan ringkasan draf silabus mata kuliah resmi"
            f"{prodi_ctx} yang paling relevan untuk menjawab pertanyaan mahasiswa berikut.\n\n"
            f"Format jawaban wajib singkat dan padat (maksimal 3-4 kalimat):\n"
            f"- Estimasi Nama Mata Kuliah Resmi:\n"
            f"- Topik Bahasan Pokok:\n"
            f"- Relevansi Kompetensi:\n\n"
            f"Pertanyaan Mahasiswa: \"{query}\"\n\n"
            f"Draf Silabus Hipotesis:"
        )

    def generate(self, query: str, prodi_hint: str | None = None) -> str:
        prompt = self.build_prompt(query=query, prodi_hint=prodi_hint)
        try:
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    "num_predict": self.max_tokens,
                    "temperature": self.temperature,
                    "stop": ["\n\n\n", "User:", "Mahasiswa:", "Pertanyaan:"],
                },
            )
            raw_text = response.get("response", "").strip()
            if not raw_text:
                return query
            return raw_text
        except Exception as e:
            logger.warning("HyDE generation failed (%s). Falling back to raw query.", e)
            return query


_GLOBAL_HYDE_GENERATOR: HydeGenerator | None = None


def get_hyde_generator(model_name: str | None = None) -> HydeGenerator:
    global _GLOBAL_HYDE_GENERATOR
    if _GLOBAL_HYDE_GENERATOR is None or (model_name and _GLOBAL_HYDE_GENERATOR.model_name != model_name):
        _GLOBAL_HYDE_GENERATOR = HydeGenerator(model_name=model_name)
    return _GLOBAL_HYDE_GENERATOR
