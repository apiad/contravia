import pytest

from contravia.rules import fleet, handling

SPEC_TABLE = {  # spec section 09: mass, power, drive, gears
    "almendron": (1600, 75, "rear", 3), "ruso": (1030, 52, "rear", 4), "moscovita": (1080, 55, "rear", 4),
    "moderno": (1250, 85, "front", 5), "huevito": (620, 18, "rear", 4), "patrulla": (1250, 100, "front", 5),
    "guagua": (11000, 180, "rear", 4), "camion": (8000, 130, "rear", 5),
}


@pytest.mark.parametrize("name", sorted(SPEC_TABLE))
def test_each_family_has_the_spec_numbers(name):
    h = handling.HANDLING[name]
    mass, power, drive, gears = SPEC_TABLE[name]
    assert (h.mass, h.power_kw, h.drive, len(h.gears)) == (mass, power, drive, gears)


def test_mass_scales_with_the_body_volume():
    preset = fleet.almendron()
    bigger = type(preset)(**{**preset.__dict__, "length": preset.length * 1.1, "width": preset.width * 1.1,
                             "height": preset.height * 1.1})
    assert handling.spec_for(bigger, "almendron").mass == pytest.approx(1600 * 1.331, rel=1e-6)
    assert handling.spec_for(preset, "almendron").mass == pytest.approx(1600)


def test_static_loads_sum_to_the_weight():
    spec = handling.spec_for(fleet.moderno(), "moderno")
    a = max(w.mount[0] for w in spec.wheels)
    b = -min(w.mount[0] for w in spec.wheels)
    loads = [spec.mass * 9.81 * (b if w.mount[0] > 0 else a) / (a + b) / 2 for w in spec.wheels]
    assert sum(loads) == pytest.approx(spec.mass * 9.81)
    assert {w.driven for w in spec.wheels if w.mount[0] > 0} == {True}


@pytest.mark.parametrize("family, expect", [("compartido", "almendron"), ("estatal", "ruso"),
                                            ("policia", "patrulla"), ("metroguagua", "guagua"),
                                            ("volquete", "camion"), ("bomberos", "camion")])
def test_every_drivable_family_finds_its_handling(family, expect):
    assert handling.handling_of(family) is handling.HANDLING[expect]


def test_cycles_are_not_driven():
    with pytest.raises(KeyError, match="bicicleta"):
        handling.handling_of("bicicleta")


def test_every_drivable_garaje_preset_has_a_spec():
    for key, preset in fleet.catalogue().items():
        if preset.drivable:
            spec = handling.spec_for(preset.params, key)
            assert spec.mass > 0, key


@pytest.mark.parametrize("key", ["metroguagua", "caja"])
def test_a_bus_and_a_truck_draw_from_the_driven_pose(key):
    from types import SimpleNamespace

    from manifoldx.modeling.assets.car import place_car
    from manifoldx.vehicles import FlatGround

    from contravia.car import CarState, Driven

    preset = fleet.catalogue()[key]
    model = fleet.BUILDERS[preset.family](preset.params)
    d = Driven(handling.spec_for(preset.params, key), CarState(0, 0, 0, 0.0), FlatGround())
    handles = {p.name: [SimpleNamespace(transform=SimpleNamespace())] for p in model.parts}
    place_car(handles, model, **d.pose())
    assert all(hasattr(h[0].transform, "pos") for h in handles.values())
