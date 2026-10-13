from src.engine.decider import DecisionCategory, System1Decider


def test_decider_initialization():
    decider = System1Decider()
    decider.initialize()
    assert decider.centroid_matrix is not None
    assert decider.centroid_matrix.shape == (6, 768)


def test_chitchat_prunes_tools():
    decider = System1Decider()
    res = decider.decide("Halo! Selamat pagi.")
    assert res.category == DecisionCategory.CHAT_ONLY
    assert res.selected_tools is None
    assert res.confidence >= 0.50


def test_prerequisite_selection():
    decider = System1Decider()
    # Explicit
    res1 = decider.decide("Apakah Data Mining I memiliki prasyarat Algoritma Pemrograman I?")
    assert res1.category == DecisionCategory.PREREQUISITE
    assert res1.selected_tools is not None
    tool_names1 = [t["function"]["name"] for t in res1.selected_tools]
    assert "validate_prerequisites_dag" in tool_names1

    # Implicit slang
    res2 = decider.decide("Kira-kira terganjal matkul lama gak ya kalau ambil ini?")
    assert res2.category == DecisionCategory.PREREQUISITE
    assert res2.selected_tools is not None


def test_sks_calculation_selection():
    decider = System1Decider()
    res = decider.decide("Berapa batas maksimal SKS jika IPS semester lalu 3.65?")
    assert res.category == DecisionCategory.SKS_CALCULATION
    assert res.selected_tools is not None
    tool_names = [t["function"]["name"] for t in res.selected_tools]
    assert "calculate_sks_limit" in tool_names
    assert len(tool_names) == 1


def test_degree_planning_selection():
    decider = System1Decider()
    res = decider.decide("Tolong buatkan rekomendasi rencana studi semester 5 Teknik Industri.")
    assert res.category == DecisionCategory.DEGREE_PLANNING
    assert res.selected_tools is not None
    tool_names = [t["function"]["name"] for t in res.selected_tools]
    assert "generate_degree_plan" in tool_names


def test_latency_under_threshold():
    decider = System1Decider()
    # Warm up
    decider.decide("Halo")
    # Test latency
    res = decider.decide("Berapa jatah SKS untuk IPS 3.2?")
    assert res.latency_ms <= 60.0
