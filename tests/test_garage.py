"""The Garaje: the fleet's catalogue and ranges, the saved designs, and its screens driven by keys."""

import json

import pytest

from contravia import garage
from contravia.rules import designs, fleet


def test_every_preset_builds_inside_the_ranges_the_garaje_offers():
    for key, preset in fleet.catalogue().items():
        d = designs.Design(key)
        assert designs.valid(d.params()) is None, key
        r = designs.street_ranges(preset.family)
        for k, spec in r.items():
            v = d.value(k)
            if "choices" in spec:
                assert v in spec["choices"], (key, k, v)
            else:
                assert spec["min"] <= float(v) <= spec["max"], (key, k, v)


def test_no_name_on_the_garaje_is_a_trademark():
    names = " ".join(p.name.lower() for p in fleet.catalogue().values())
    for mark in ("lada", "moskvich", "polski", "fiat", "chevrolet", "ford", "kamaz", "zil", "yutong"):
        assert mark not in names


def test_each_numeric_knob_at_either_end_of_its_live_limits_is_valid():
    for key, preset in fleet.catalogue().items():
        d = designs.Design(key)
        for k, spec in designs.street_ranges(preset.family).items():
            if "choices" in spec:
                continue
            lo, hi = designs.limits(d, k)
            assert spec["min"] <= lo <= hi <= spec["max"]
            for v in (lo, hi):
                assert designs.valid(designs._set(d, k, v).params()) is None, (key, k, v)


def test_turning_a_knob_stops_at_its_end_and_says_so():
    d = designs.Design("ruso")
    why = ""
    for _ in range(200):
        d, why = designs.turn(d, "length", 1)
        if why:
            break
    assert d.value("length") == pytest.approx(designs.street_ranges("car")["length"]["max"])
    assert why


def test_turning_an_option_goes_round_its_choices():
    d = designs.Design("almendron")
    seen = set()
    for _ in range(len(designs.street_ranges("car")["headlamp"]["choices"])):
        d, _ = designs.turn(d, "headlamp", 1)
        seen.add(d.value("headlamp"))
    assert seen == set(designs.street_ranges("car")["headlamp"]["choices"])


def test_fix_pulls_a_design_back_inside_and_leaves_a_valid_one_alone():
    d = designs.Design("ruso", changes={"length": 9.0, "headlamp": "laser", "nope": 1}, paint="#123456")
    f = designs.fix(d)
    assert f.changes == {"length": designs.street_ranges("car")["length"]["max"]}
    assert f.paint is None and designs.valid(f.params()) is None
    ok = designs.Design("ruso", changes={"length": 4.4}, paint=designs.COLOURS[0][1], livery="Franja")
    assert designs.fix(ok) == ok


def test_a_livery_paints_the_side_with_the_second_colour():
    d = designs.Design("ruso", livery="Policía", second="#2f5fa8")
    p = d.params()
    assert p.lightbar == "bar" and p.livery[0][0] == "#2f5fa8"
    assert d.palette()["roof"][0] == "#2f5fa8"


def test_designs_are_saved_read_back_and_fixed_on_load(tmp_path, monkeypatch):
    monkeypatch.setenv("CONTRAVIA_CONFIG", str(tmp_path))
    d = designs.Design("almendron", "Mi almendrón 1", {"length": 5.0}, paint=designs.COLOURS[4][1])
    garage.save_design(d)
    assert garage.load_designs() == [d]
    path = tmp_path / "garaje" / "mi-almendron-1.json"
    data = json.loads(path.read_text())
    data["changes"]["length"] = 99.0  # a range tightened after it was saved
    path.write_text(json.dumps(data))
    (back,) = garage.load_designs()
    assert back.changes["length"] == designs.street_ranges("car")["length"]["max"]
    assert garage.new_name("Almendrón", ["mi almendrón 1"]) == "Mi almendrón 1"
    assert garage.new_name("Almendrón", ["Mi almendrón 1"]) == "Mi almendrón 2"
    garage.delete_design(back)
    assert garage.load_designs() == []


def test_colours_do_not_rebuild_the_geometry():
    a = designs.Design("moto", paint=designs.COLOURS[0][1])
    b = designs.Design("moto", paint=designs.COLOURS[1][1])
    assert garage.key(a) == garage.key(b)
    assert garage.key(a) != garage.key(designs.Design("moto", changes={"wheelbase": 1.5}))


def test_the_garaje_browses_designs_saves_and_drives(city_npz, tmp_path, monkeypatch):
    import manifoldx as mx

    from contravia.clock import Clock, parse_at
    from contravia.game import Game
    from contravia.menu import Menu
    from contravia.settings import FreeDrive, Options
    from contravia.world import World

    monkeypatch.setenv("CONTRAVIA_CONFIG", str(tmp_path))
    engine = mx.Engine("garaje", width=64, height=48)
    game = Game(engine, World.load(str(city_npz)), clock=Clock(at=parse_at("Tue 10:00")))
    menu = Menu(engine, game, FreeDrive.from_flags(when="Tue 10:00"), Options.load())
    engine.render_frame(supersample=1)
    for _ in range(2):
        menu.press("ArrowDown")  # from Modo libre to Garaje
    menu.press("Enter")
    assert menu.current == "garage" and game.mode == "garage"
    salon = menu.screens["garage"]
    assert salon.tab == "Carros" and [d.base for d in salon.items][:3] == ["almendron", "ruso", "moscovita"]
    menu.press("ArrowDown")  # Ruso
    assert menu.target.base == "ruso"
    menu.press("e")  # Guaguas
    assert menu.current == "garage" and menu.screens["garage"].tab == "Guaguas"
    menu.press("q")
    menu.press("ArrowDown")
    menu.press("Tab")
    assert menu.current == "design" and menu.design.base == "ruso"
    before = menu.design.value("length")
    menu.press("ArrowRight")  # Largo
    assert menu.design.value("length") > before and not menu.saved
    menu.press("z")
    assert menu.design.value("length") == before
    menu.press("ArrowRight")
    menu.press("g")
    assert menu.saved and [d.name for d in menu.designs] == ["Mi ruso 1"]
    assert (tmp_path / "garaje" / "mi-ruso-1.json").exists()
    menu.press("Enter")  # drive it: through the free drive's screen, the design in its Vehículo row
    assert menu.current == "free" and menu.free["vehicle"].text == "Mi ruso 1"
    assert game.model is not menu.default_model and game.photo is None
    menu.press("Enter")  # Arrancar
    for _ in range(600):  # the drive starts under the loading curtain, which lifts once the start is drawn
        if game.curtain is None:
            break
        engine.render_frame(supersample=1, dt=0.05)
    assert menu.current is None and game.mode == "drive"
    menu.press("Escape")
    for _ in range(2):
        menu.press("ArrowDown")  # Cambiar vehículo
    menu.press("Enter")
    assert menu.current == "garage" and game.mode == "garage"
    menu.press("Escape")  # back to the pause, the design still driven
    assert menu.current == "pause" and menu.chosen.name == "Mi ruso 1"
