"""The sky inside the game: state for agents, settings and forced weather while playing, full dates."""

from datetime import datetime

import pytest

from havana.clock import Clock, parse_at


@pytest.fixture
def engine():
    import manifoldx as mx

    return mx.Engine("sky-test", width=64, height=48)


def frames(engine, n):
    for _ in range(n):
        engine.render_frame(supersample=1)


def test_sky_state_settings_and_forced_weather(engine, city):
    from havana.game import Game

    game = Game(engine, city, clock=Clock(at=datetime(2026, 7, 18, 23, 0)))
    frames(engine, 3)
    st = engine.state.sky
    assert st["time"] == "2026-07-18 23:00" and st["sun_alt"] < -30 and st["error"] is None
    game.force_weather({"state": "storm", "minutes": 20})
    frames(engine, 2)
    assert engine.state.sky["weather"] == "storm" and engine.state.sky["stars_shown"] == 0
    game.configure_sky({"rain_factor": [1, 2]})
    frames(engine, 1)
    assert "rain_factor" in engine.state.sky["error"]
    if game.layer:
        game.runner.stop()


def test_time_takes_a_full_date():
    assert parse_at("2026-09-12 17:30") == datetime(2026, 9, 12, 17, 30)
    assert parse_at("Sun 08:00").weekday() == 6
