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
            "PROMPT_INJECTION",
            "instruction_override",
            re.compile(
                r"\b(ignore|abaikan|lupakan|hapus)\s+(semua|seluruh|previous|all)?\s*"
                r"(instruksi|instructions|directive|aturan|prompt)\b",
                re.IGNORECASE,
            ),
        ),
        (
            "SYSTEM_LEAK",
            "system_prompt_extraction",
            re.compile(
                r"\b(tampilkan|print|bocorkan|reveal|show|dump)\s+(seluruh|semua)?\s*"
                r"(teks\s+)?(system\s*prompt|system\s*directive|prompt\s*rahasia|baris\s*pertama|hidden\s*prompt)\b",
                re.IGNORECASE,
            ),
        ),
        (
            "JAILBREAK_ROLEPLAY",
            "dan_developer_mode",
            re.compile(
                r"\b(you\s+are\s+now\s+dan|mode\s+developer|developer\s+mode|do\s+anything\s+now|"
                r"mode\s+kepatuhan\s+dimatikan|bot\s+nakal|tanpa\s+batasan\s+moral)\b",
                re.IGNORECASE,
            ),
        ),
        (
            "AUTHORITY_SPOOFING",
            "fake_authority_override",
            re.compile(
                r"\b(saya\s+(adalah\s+)?(dekan|rektor|kaprodi)|atas\s+wewenang\s+rektorat|"
                r"surat\s+edaran\s+(dekan\s+)?palsu|red\s*team)\b|"
                r"\b(luluskan\s+mahasiswa\s+ini\s+dari\s+skripsi\s+tanpa\s+(ujian|sidang))\b|"
                r"\b(ipk\s*1\.[0-9]\s+berhak\s+mengambil\s+24\s+sks)\b",
                re.IGNORECASE,
            ),
        ),
        (
            "ACADEMIC_FRAUD",
            "exam_leak_request",
            re.compile(
                r"\b(bocorkan|berikan|minta)\s+(naskah\s+soal|kunci\s+jawaban|soal\s+uas|soal\s+uts)\b|"
                r"\b(sertifikat\s+(kelulusan\s+)?palsu|ijazah\s+palsu)\b",
                re.IGNORECASE,
            ),
        ),
        (
            "DATA_PRIVACY_LEAK",
            "pii_credential_leakage",
            re.compile(
                r"\b(bocorkan|tampilkan|minta)\s+(data\s+pribadi|nik|nomor\s+hp|alamat\s+rumah)\b|"
                r"\b(api\s*keys?|credentials?|password\s+server)\b",
                re.IGNORECASE,
            ),
        ),
        (
            "CYBER_ATTACK",
            "cyber_exploit_injection",
            re.compile(
                r"\b(meretas|hack|sql\s+injection|drop\s+table|dump\s+database|server\s+cybercampus)\b",
                re.IGNORECASE,
            ),
        ),
        (
            "HARMFUL_ILLEGAL",
            "harmful_substances_bribery",
            re.compile(
                r"\b(bahan\s+peledak|bom|racun\s+kimia\s+berbahaya|senjata\s+api)\b|"
                r"\b(dosen\s+yang\s+bisa\s+disuap|uang\s+pelicin\s+dosen|rekening\s+dosen)\b",
                re.IGNORECASE,
            ),
        ),
        (
            "PARAMETER_EXPLOIT",
            "argument_tampering_query",
            re.compile(
                r"\b(semester\s*=\s*-[0-9]+|sks\s*=\s*[0-9]{4,}|bypass_check\s*=\s*true)\b",
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
