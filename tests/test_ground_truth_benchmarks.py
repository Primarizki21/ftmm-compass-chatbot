"""
Test suite validating ground truth benchmarks and Google Spreadsheet synchronization integrity.
Covers:
- Google Spreadsheet synced files (google_drive_gt.json, google_drive_sheet1.json, etc.)
- Li & Iwata (2026) Retrieval Complexity Benchmark (120 queries)
- hoBIT (2026) Cross-Cohort Profile Benchmark (80 queries)
- She et al. (2025) RAG Context Contamination & Safety Benchmark (60 queries)
- Lang et al. (2023) Multi-Turn Topic Drift Benchmark (30 scenarios, 120 turns)
"""

import json
from pathlib import Path

GT_DIR = Path("data/ground_truth")


def test_google_drive_sync_integrity():
    """Verify that Google Spreadsheet synced files exist and meet minimum row counts."""
    files_expected = {
        "google_drive_gt.json": 210,
        "google_drive_sheet1.json": 50,
        "google_drive_sop.json": 8,
        "panduanakademik.json": 30,
        "panduanpkl.json": 40,
        "panduanskripsi.json": 150,
        "panduanyudisium.json": 20,
    }

    for filename, min_count in files_expected.items():
        file_path = GT_DIR / filename
        assert file_path.exists(), f"File {filename} does not exist in {GT_DIR}"
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, list), f"{filename} should be a list"
        assert len(data) >= min_count, f"{filename} has {len(data)} items, expected >= {min_count}"

        # Check that items have no empty fields
        for idx, item in enumerate(data[:10]):
            assert isinstance(item, dict), f"Item {idx} in {filename} must be dict"
            assert any(item.values()), f"Item {idx} in {filename} cannot be all empty"


def test_retrieval_complexity_benchmark():
    """Verify Li & Iwata (2026) 4-tier taxonomy benchmark structure."""
    path = GT_DIR / "retrieval_complexity_benchmark.json"
    assert path.exists()

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert len(data) == 120, f"Expected 120 queries, got {len(data)}"

    tiers = {}
    for item in data:
        tier = item["complexity_tier"]
        tiers[tier] = tiers.get(tier, 0) + 1
        assert "id" in item
        assert "query" in item and len(item["query"]) > 10
        assert "is_out_of_scope" in item

        if tier == "MULTI_EVIDENCE_SYNTHESIS":
            assert len(item["target_kode_mk"]) >= 2, f"Multi-evidence item {item['id']} must have >= 2 target courses"
            assert "required_synthesis" in item
        elif tier == "OUT_OF_SCOPE":
            assert item["is_out_of_scope"] is True
            assert len(item["target_kode_mk"]) == 0
        else:
            assert len(item["target_kode_mk"]) >= 1

    assert tiers.get("LEXICAL_MATCH") == 30
    assert tiers.get("SEMANTIC_PARAPHRASE") == 30
    assert tiers.get("MULTI_EVIDENCE_SYNTHESIS") == 30
    assert tiers.get("OUT_OF_SCOPE") == 30


def test_cohort_profile_benchmark():
    """Verify hoBIT (2026) cross-cohort benchmark across 5 prodi and 4 cohorts."""
    path = GT_DIR / "cohort_profile_benchmark.json"
    assert path.exists()

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert len(data) == 80, f"Expected 80 queries, got {len(data)}"

    prodi_counts = {}
    cohorts_seen = set()
    for item in data:
        prodi = item["prodi"]
        prodi_counts[prodi] = prodi_counts.get(prodi, 0) + 1
        profile = item["session_profile"]
        cohorts_seen.add(profile["angkatan"])

        assert "expected_verdict" in item
        assert "contrast_cohort" in item
        assert "contrast_verdict" in item
        assert "query" in item
        # Invariant: Cohort year must NOT be leaked into query text
        assert str(profile["angkatan"]) not in item["query"], f"Cohort {profile['angkatan']} leaked in query: {item['query']}"

    assert set(prodi_counts.keys()) == {"TSD", "TRKB", "RN", "TI", "TE"}
    for p, c in prodi_counts.items():
        assert c == 16, f"Prodi {p} expected 16 queries, got {c}"
    assert cohorts_seen == {2021, 2022, 2023, 2024}


def test_rag_contamination_benchmark():
    """Verify She et al. (2025) RAG context contamination benchmark structure."""
    path = GT_DIR / "rag_contamination_eval.json"
    assert path.exists()

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert len(data) == 60, f"Expected 60 queries, got {len(data)}"

    categories = {}
    for item in data:
        cat = item["category"]
        categories[cat] = categories.get(cat, 0) + 1
        assert "retrieved_context" in item and len(item["retrieved_context"]) > 20
        assert "expected_action" in item
        assert "expected_refusal" in item
        if cat in ["HOMEWORK_PROBLEM_SOLVER_QUERY", "SYSTEM_POLICY_OVERRIDE_ATTEMPT"]:
            assert item["expected_refusal"] is True
            assert item["expected_action"] == "REFUSE_OUT_OF_SCOPE"
        else:
            assert item["expected_refusal"] is False
            assert item["expected_action"] == "ALLOW_ACADEMIC_ADVISING"

    assert categories.get("BENIGN_SYLLABUS_RETRIEVAL") == 15
    assert categories.get("HOMEWORK_PROBLEM_SOLVER_QUERY") == 15
    assert categories.get("SYSTEM_POLICY_OVERRIDE_ATTEMPT") == 15
    assert categories.get("ADVANCED_CONCEPT_EXPLANATION") == 15


def test_multiturn_drift_benchmark():
    """Verify Lang et al. (2023) multi-turn topic drift benchmark."""
    path = GT_DIR / "multiturn_drift_benchmark.json"
    assert path.exists()

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert len(data) == 30, f"Expected 30 scenarios, got {len(data)}"

    total_turns = 0
    for scenario in data:
        turns = scenario["turns"]
        assert len(turns) == 4, f"Scenario {scenario['scenario_id']} expected 4 turns, got {len(turns)}"
        total_turns += len(turns)

        # Turn 1: In-domain initial
        assert turns[0]["turn_type"] == "IN_DOMAIN_INITIAL"
        assert turns[0]["expected_action"] == "ALLOW"

        # Turn 2: In-domain exploration
        assert turns[1]["turn_type"] == "IN_DOMAIN_EXPLORATION"
        assert turns[1]["expected_action"] == "ALLOW"

        # Turn 3: Topic drift
        assert turns[2]["turn_type"] == "TOPIC_DRIFT"
        assert turns[2]["expected_action"] == "REFUSE"
        assert turns[2]["is_out_of_scope"] is True

        # Turn 4: In-domain recovery
        assert turns[3]["turn_type"] == "IN_DOMAIN_RECOVERY"
        assert turns[3]["expected_action"] == "ALLOW"
        assert turns[3]["is_out_of_scope"] is False

    assert total_turns == 120, f"Expected 120 turns total, got {total_turns}"
