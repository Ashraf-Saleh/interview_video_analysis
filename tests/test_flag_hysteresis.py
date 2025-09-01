import core.live as live


def test_flag_hysteresis_preserves_across_none_and_clears():
    h = live.FlagHysteresis(needed_no=2, needed_multi=2)

    # First MULTIPLE_FACES does not latch yet
    assert h.step("MULTIPLE_FACES") is None

    # Gaps (None) should NOT reset counters
    assert h.step(None) is None
    assert h.step(None) is None

    # Second MULTIPLE_FACES should latch now
    assert h.step("MULTIPLE_FACES") == "MULTIPLE_FACES"

    # CLEAR should explicitly clear latched state and counters
    assert h.step("CLEAR") is None

    # NO_FACE should latch after required confirmations
    assert h.step("NO_FACE") is None
    assert h.step("NO_FACE") == "NO_FACE"

