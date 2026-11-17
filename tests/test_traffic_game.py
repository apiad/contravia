from datetime import datetime

import pytest

from contravia.citydata import CityData
from contravia.clock import Clock

TUE_8 = datetime(2026, 11, 17, 8, 30)


@pytest.fixture
def engine():
    import manifoldx as mx

    return mx.Engine("traffic-test", width=64, height=48)


def frames(engine, n):
    for _ in range(n):
        engine.render_frame(supersample=1)


def test_cars_appear_and_settings_change_live(engine, city):
    from contravia.game import Game

    game = Game(engine, city, clock=Clock(at=TUE_8), with_traffic=True)
    frames(engine, 40)
    st = engine.state.traffic
    assert st["error"] is None and st["cars"] > 0
    before = st["moving_now"]["car"]
    game.configure_traffic({"population": str(2 * game.traffic_config.population)})
    frames(engine, 2)
    assert engine.state.traffic["moving_now"]["car"] == pytest.approx(2 * before, rel=0.01)
    game.runner.stop()


def test_bad_settings_are_refused_with_a_reason(engine, city):
    from contravia.game import Game

    game = Game(engine, city, clock=Clock(at=TUE_8), with_traffic=True)
    old = game.traffic_config
    game.configure_traffic({"populaton": 5})
    assert "populaton" in engine.state.traffic["error"]
    assert game.traffic_config is old
    game.runner.stop()


def test_data_without_a_graph_plays_without_traffic(engine, city):
    from contravia.game import Game

    old = CityData({k: v for k, v in city.a.items() if not k.startswith("g_")})
    Game(engine, old, clock=Clock(at=TUE_8), with_traffic=True)
    frames(engine, 5)
    assert "road graph" in engine.state.traffic["error"]


def test_traffic_is_off_unless_asked(engine, city):
    from contravia.game import Game

    game = Game(engine, city, clock=Clock(at=TUE_8))
    frames(engine, 5)
    assert game.layer is None and engine.state.traffic == {"error": None, "on": False}
