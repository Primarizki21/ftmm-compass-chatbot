import pytest
from src.engine.normalizer import (
    AcademicNormalizer,
    damerau_levenshtein_distance,
    get_academic_normalizer,
    normalize_query,
)


@pytest.fixture(scope="module")
def normalizer():
    return get_academic_normalizer()


def test_damerau_levenshtein_distance():
    # Exact match
    assert damerau_levenshtein_distance("algoritma", "algoritma") == 0
    # Insertion
    assert damerau_levenshtein_distance("algritma", "algoritma") == 1
    # Deletion
    assert damerau_levenshtein_distance("algoriitma", "algoritma") == 1
    # Substitution
    assert damerau_levenshtein_distance("algorirma", "algoritma") == 1
    # Transposition
    assert damerau_levenshtein_distance("algortima", "algoritma") == 1
    assert damerau_levenshtein_distance("semseter", "semester") == 1
    assert damerau_levenshtein_distance("strutkur", "struktur") == 1
    # Boundary max distance
    assert damerau_levenshtein_distance("kucing", "algoritma", max_dist=2) > 2
    # Empty strings
    assert damerau_levenshtein_distance("", "") == 0
    assert damerau_levenshtein_distance("abc", "") == 3
    assert damerau_levenshtein_distance("", "def") == 3


def test_slang_lexicon_normalization(normalizer):
    cases = [
        ("apa syarat matkul strukdat?", "Struktur Data"),
        ("silabus alpro semester 2", "Algoritma Pemrograman"),
        ("info metpen di semester 5", "Metodologi Penelitian"),
        ("jadwal jarkom dan sisop", "Jaringan Komputer"),
        ("apakah basdat ada praktikum?", "Basis Data"),
        ("makul pbo dan rpl", "Pemrograman Berorientasi Objek"),
        ("kalku 1 prasyarat kalku 2", "Kalkulus"),
        ("matdis dan komnum di smt 3", "Matematika Diskrit"),
        ("anreg di prodi tsd", "Analisis Regresi"),
        ("elka dan sisdig di te", "Elektronika"),
    ]
    for raw, expected_substring in cases:
        res = normalizer.normalize(raw)
        assert expected_substring in res.normalized_query, (
            f"Expected '{expected_substring}' in '{res.normalized_query}' for query '{raw}'"
        )


def test_stt_phonetic_normalization(normalizer):
    cases = [
        ("info struktur dada semester 3", "Struktur Data"),
        ("silabus argo ritme dan pemprograman", "Algoritma"),
        ("mata kulia pasis data", "Basis Data"),
        ("bagaimana cara daftar pekael", "Praktik Kerja Lapangan"),
        ("syarat ambil skipsi", "skripsi"),
        ("apakah semeter ganjil sudah mulai?", "semester"),
        ("kecerdasan buwatan di prodi trkb", "Kecerdasan Buatan"),
        ("rekayasa perankat lunak", "Rekayasa Perangkat Lunak"),
    ]
    for raw, expected_substring in cases:
        res = normalizer.normalize(raw)
        assert expected_substring.lower() in res.normalized_query.lower(), (
            f"Expected '{expected_substring}' in '{res.normalized_query}' for query '{raw}'"
        )


def test_levenshtein_typo_correction(normalizer):
    cases = [
        ("silabus algortima dan pemrogaman", "algoritma"),
        ("silabus algortima dan pemrogaman", "pemrograman"),
        ("klakulus 1 ada di semester berapa?", "kalkulus"),
        ("apa prsyarat matkul ini?", "prasyarat"),
        ("syarat ambil skirpsi di semester 7", "skripsi"),
        ("syarat yudisum apa saja?", "yudisium"),
    ]
    for raw, expected_token in cases:
        res = normalizer.normalize(raw)
        assert expected_token in res.normalized_query.lower(), (
            f"Expected corrected token '{expected_token}' in '{res.normalized_query}' for query '{raw}'"
        )


def test_course_code_correction(normalizer):
    res1 = normalizer.normalize("apakah matkul SIC2O3 ada prasyaratnya?")
    assert "SIC203" in res1.normalized_query

    res2 = normalizer.normalize("info silabus AGI1O1 prodi TSD")
    assert "AGI101" in res2.normalized_query

    # Pastikan TSD tidak diubah menjadi TID
    res3 = normalizer.normalize("daftar matkul semester 1 TSD")
    assert "Teknologi Sains Data" in res3.normalized_query or "TSD" in res3.normalized_query
    assert "TID" not in res3.normalized_query


def test_conversational_stopwords_preservation(normalizer):
    # Kata pendek dan stopwords tidak boleh dirusak typo matcher
    raw = "apa bisa saya ambil mata kuliah ini untuk semester depan?"
    res = normalizer.normalize(raw)
    words = res.normalized_query.split()
    assert "apa" in words
    assert "bisa" in words
    assert "saya" in words
    assert "mata" in words
    assert "kuliah" in words
    assert "tata" not in words  # Invariant: 'mata' never becomes 'tata'


def test_edge_cases(normalizer):
    # Empty query
    res_empty = normalizer.normalize("")
    assert res_empty.normalized_query == ""
    assert res_empty.traces == []

    # Whitespace only
    res_ws = normalizer.normalize("    \n\t  ")
    assert res_ws.normalized_query == ""

    # Punctuation preservation
    res_punc = normalizer.normalize("apa prsyarat matkul strukdat?!")
    assert res_punc.normalized_query.endswith("?!")
    assert "prasyarat" in res_punc.normalized_query

    # Helper function
    helper_out = normalize_query("matkul alpro")
    assert "mata kuliah Algoritma Pemrograman" in helper_out
