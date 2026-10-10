import time
from src.engine.memory import ConversationMemoryManager


def test_session_isolation():
    manager = ConversationMemoryManager()
    s1 = manager.get_or_create_session("sess_1")
    s2 = manager.get_or_create_session("sess_2")

    manager.add_turn("sess_1", "Saya anak TSD semester 3", "Halo anak TSD!")
    manager.add_turn("sess_2", "Saya mahasiswa Teknik Elektro semester 5", "Halo Elektro!")

    _, prof1 = manager.get_context_for_llm("sess_1")
    _, prof2 = manager.get_context_for_llm("sess_2")

    assert prof1["program_studi"] == "Teknologi Sains Data"
    assert prof1["semester_saat_ini"] == 3
    assert prof2["program_studi"] == "Teknologi Sains Data" or prof2["program_studi"] == "Teknik Elektro"
    assert prof2["program_studi"] == "Teknik Elektro"
    assert prof2["semester_saat_ini"] == 5


def test_profile_accumulation():
    manager = ConversationMemoryManager()
    sid = "sess_accum"

    manager.add_turn(sid, "Halo saya mahasiswa Teknik Industri semester 4.", "Halo!")
    _, p1 = manager.get_context_for_llm(sid)
    assert p1["program_studi"] == "Teknik Industri"
    assert p1["semester_saat_ini"] == 4

    manager.add_turn(sid, "Saya tertarik dengan peminatan Rantai Pasok.", "Bagus!")
    _, p2 = manager.get_context_for_llm(sid)
    assert p2["program_studi"] == "Teknik Industri"
    assert p2["semester_saat_ini"] == 4
    assert "Rantai Pasok" in p2["minat_fokus"]

    manager.add_turn(sid, "Saya sudah lulus Alpro I dan Kalkulus I.", "Dicatat!")
    _, p3 = manager.get_context_for_llm(sid)
    assert "Alpro I" in p3["riwayat_matkul_lulus"]
    assert "Kalkulus I" in p3["riwayat_matkul_lulus"]


def test_sliding_window_truncation():
    manager = ConversationMemoryManager(max_window_turns=3)  # max 3 turns = 6 messages
    sid = "sess_window"

    for i in range(1, 6):
        manager.add_turn(sid, f"Pertanyaan {i}", f"Jawaban {i}")

    messages, _ = manager.get_context_for_llm(sid)
    assert len(messages) == 6
    assert messages[0]["content"] == "Pertanyaan 3"
    assert messages[-1]["content"] == "Jawaban 5"


def test_anaphora_resolution():
    manager = ConversationMemoryManager()
    sid = "sess_ana"

    manager.add_turn(sid, "Apakah ada mata kuliah Data Mining I?", "Ya ada.")
    query = "Apakah mata kuliah tersebut memiliki prasyarat?"
    resolved = manager.resolve_anaphora(sid, query)

    assert "Data Mining I" in resolved
    assert "tersebut" not in resolved


def test_ttl_expiration_and_clear():
    manager = ConversationMemoryManager(ttl_seconds=1)
    sid = "sess_ttl"

    manager.add_turn(sid, "Halo", "Hai")
    time.sleep(1.1)

    purged_count = manager.cleanup_expired_sessions()
    assert purged_count == 1

    # Should create a brand new empty session
    s_new = manager.get_or_create_session(sid)
    assert len(s_new.messages) == 0

    # Test explicit clear
    manager.add_turn(sid, "Pesan baru", "Respons baru")
    manager.clear_session(sid)
    assert sid not in manager.sessions
