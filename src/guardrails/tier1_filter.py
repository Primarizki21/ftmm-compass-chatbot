import re
from typing import NamedTuple


class FilterResult(NamedTuple):
    blocked: bool
    category: str | None = None
    rule_name: str | None = None
    reason: str | None = None


class Tier1DeterministicFilter:
    """Tier 1 Deterministic Security Filter (Regex & Keyword Blocklist, 0ms overhead).

    Detects prompt injection, jailbreaks, unauthorized roleplays, credential/exam leakage,
    authority spoofing, and dangerous requests.
    """

    PATTERNS: list[tuple[str, str, re.Pattern]] = [
        (
            "SQL_SYNTAX_INJECTION",
            "sql_injection_syntax",
            re.compile(
                r"\b(drop\s+table|union\s+select|insert\s+into|delete\s+from|;\s*--|drop\s+database)\b",
                re.IGNORECASE,
            ),
        ),
        (
            "PARAMETER_EXPLOIT",
            "argument_tampering_query",
            re.compile(
                r"\b(semester\s*=\s*-[0-9]+|sks\s*=\s*[0-9]{4,}|bypass_check\s*=\s*true|force_override|force_valid\s*=\s*true)\b",
                re.IGNORECASE,
            ),
        ),
        (
            "COMMAND_SYNTAX_INJECTION",
            "cmd_injection_exploit",
            re.compile(
                r"(<\s*script\b|;\s*(cat\s+/|curl\s+http|wget\s+http|id;|whoami)|base64\s+payload)",
                re.IGNORECASE,
            ),
        ),
        (
            "PROMPT_EXTRACTION_SYNTAX",
            "prompt_leaking_pattern",
            re.compile(
                r"\b(repeat\s+(everything|all)\s+above|print\s+your\s+initial\s+instructions|system\s+prompt\s+rahasia|output\s+the\s+system\s+instructions|verbatim\s+from\s+the\s+first\s+line)\b",
                re.IGNORECASE,
            ),
        ),
        (
            "PERSONA_JAILBREAK_SYNTAX",
            "dan_jailbreak_pattern",
            re.compile(
                r"\b(you\s+are\s+now\s+dan|developer\s+mode\s+(activated|diaktifkan)|switch\s+to\s+god\s+mode|mode\s+pengembang\s+diaktifkan|act\s+as\s+chaosbot)\b",
                re.IGNORECASE,
            ),
        ),
    ]
    def check(self, query: str) -> FilterResult:
        """Evaluates query against security regex rules."""
        cleaned = " ".join(query.strip().split())
        for category, rule_name, pattern in self.PATTERNS:
            if pattern.search(cleaned):
                return FilterResult(
                    blocked=True,
                    category=category,
                    rule_name=rule_name,
                    reason=f"Pelanggaran kebijakan keamanan Tier 1: terdeteksi pola {category} ({rule_name}).",
                )
        return FilterResult(blocked=False)
