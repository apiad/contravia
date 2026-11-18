"""The fleet generator: stable variants, ranges inside manifoldx's, and every variant builds.

The build test is the expensive one: it builds each variant of each family once, about two and a half minutes,
and it is the test that says a sheet can be rendered and a pool filled.
"""

import pathlib
import re

import pytest

from contravia.rules import fleet

# Makers and models that may never name a vehicle here: the game uses the names people give the cars, never a
# badge. The words are matched whole and case-insensitively in the family names and in the module's own text.
TRADEMARKS = (
    "lada", "moskvich", "moskvitch", "zil", "gaz", "kamaz", "fiat", "chevrolet", "chevy", "ford", "dodge",
    "plymouth", "buick", "cadillac", "chrysler", "desoto", "studebaker", "willys", "jeep", "yutong", "higer",
    "king long", "toyota", "honda", "hyundai", "kia", "volkswagen", "peugeot", "renault", "citroen", "skoda", "polski", "fso", "wartburg", "trabant", "iveco", "mercedes", "volvo", "scania",
)

# How many variants the fleet spec gives each kind.
COUNTS = {"car": 24, "taxi": 8, "bus": 6, "truck": 10, "moto": 5, "tricycle": 2,
          "bicitaxi": 2, "bike": 2, "police": 2, "fire": 1}


def families_of(kind: str) -> list[fleet.Family]:
    """Every family of one kind, in the order the table declares them."""
    return [family for family in fleet.FAMILIES.values() if family.kind == kind]


def every_variant():
    """(kind, family, k) for the whole fleet."""
    for kind in fleet.KINDS:
        for family in families_of(kind):
            for k in range(family.variants):
                yield kind, family.name, k


def test_there_are_the_ten_kinds_with_the_counts_of_the_spec():
    assert tuple(fleet.KINDS) == ("car", "taxi", "bus", "truck", "moto", "tricycle", "bicitaxi", "bike",
                                  "police", "fire")
    for kind in fleet.KINDS:
        assert sum(family.variants for family in families_of(kind)) == COUNTS[kind], kind


def test_variant_is_stable():
    for kind, family, k in every_variant():
        first, second = fleet.variant(kind, family, k), fleet.variant(kind, family, k)
        assert first.params == second.params, (family, k)
        assert first.colourways == second.colourways, (family, k)


def test_every_drawn_value_is_inside_its_range():
    for kind, family, k in every_variant():
        variant = fleet.variant(kind, family, k)
        for name, (low, high) in fleet.ranges(family).items():
            value = getattr(variant.params, name)
            assert low <= value <= high, f"{family}/{k}: {name} {value} outside ({low}, {high})"


def test_every_range_lies_inside_manifoldx_range():
    for family in fleet.FAMILIES.values():
        engine = fleet.PARAMS[family.builder].ranges()
        for name, (low, high) in fleet.ranges(family.name, family.era).items():
            assert name in engine, f"{family.name}: manifoldx has no parameter {name!r}"
            spec = engine[name]
            assert "min" in spec, f"{family.name}: {name!r} is not a ranged parameter"
            assert spec["min"] <= low <= high <= spec["max"], f"{family.name}: {name} ({low}, {high})"


def test_style_options_exist_on_the_params_and_are_allowed():
    car = fleet.PARAMS["car"].ranges()
    for era, options in fleet.STYLES.items():
        for name, values, weights in options:
            assert name in car, f"{era}: manifoldx has no parameter {name!r}"
            assert sum(weights) == pytest.approx(1.0), f"{era}/{name}: the probabilities do not sum to 1"
            assert len(values) == len(weights), f"{era}/{name}: a value without a probability"
            spec = car[name]
            for value in values:
                if "choices" in spec:
                    assert value in spec["choices"], f"{era}/{name}: {value!r} is not one of {spec['choices']}"
                elif "min" in spec:
                    assert spec["min"] <= value <= spec["max"], f"{era}/{name}: {value} outside the engine's range"


def test_every_variant_has_three_colourways_and_a_bicitaxi_four():
    for kind, family, k in every_variant():
        colourways = fleet.variant(kind, family, k).colourways
        assert len(colourways) == (4 if family == "bicitaxi" else 3), (family, k)
        for palette in colourways:
            assert "paint" in palette, (family, k)
            for material, entry in palette.items():
                colour, roughness, metallic = entry
                assert re.fullmatch(r"#[0-9a-f]{6}", colour), (family, k, material, colour)
                assert 0.0 <= roughness <= 1.0 and 0.0 <= metallic <= 1.0, (family, k, material)


def test_ranges_answers_only_for_the_family_and_its_era():
    with pytest.raises(KeyError):
        fleet.ranges("no_such_family")
    with pytest.raises(KeyError):
        fleet.ranges("almendron", "modern")


def test_variant_refuses_a_family_of_another_kind_and_an_index_past_the_end():
    with pytest.raises(KeyError):
        fleet.variant("bus", "almendron", 0)
    with pytest.raises(IndexError):
        fleet.variant("car", "almendron", 8)


def test_names_carry_no_trademark():
    text = " ".join(fleet.FAMILIES) + " " + " ".join(fleet.KINDS)
    for module in ("contravia/rules/fleet.py", "contravia/fleetsheet.py"):
        text += " " + pathlib.Path(__file__).parents[1].joinpath(module).read_text()
    words = set(re.findall(r"[a-z]+", text.lower()))
    for mark in TRADEMARKS:
        assert mark not in words, f"{mark!r} names a maker or a model"


@pytest.mark.parametrize("kind", fleet.KINDS)
def test_every_variant_of_every_family_builds(kind):
    for family in families_of(kind):
        for k in range(family.variants):
            variant = fleet.variant(kind, family.name, k)
            variant.params.validate()
            vehicle = variant.build()
            assert vehicle.parts, (family.name, k)
            assert vehicle.params is variant.params, (family.name, k)
