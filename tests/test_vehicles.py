import time

import numpy as np
import pytest

from caiman.rules.traffic import KINDS
from caiman.traffic import LENGTH
from caiman.vehicles import PARKED, merged, slot_transforms, slots


def test_one_mesh_per_kind_of_the_right_length():
    for kind in KINDS:
        m = merged(kind)
        span = m["positions"][:, 0].max() - m["positions"][:, 0].min()
        assert span == pytest.approx(LENGTH[kind], rel=0.1), kind
        assert len(m["indices"]) // 3 < 4000, kind
        assert m["colors"].shape == m["positions"].shape


def test_cars_fill_their_kind_slots_and_overflow_is_counted():
    pool = {"car": 2, "taxi": 1, "bus": 0, "moto": 0, "truck": 0}
    k = slots(pool)
    xyz = np.array([[1.0, 0, 0], [2.0, 0, 0], [3.0, 0, 0], [4.0, 0, 0]])
    kind = np.array([0, 0, 0, 1])  # three cars for two car slots, one taxi
    pos, rot, overflow = slot_transforms(k, xyz, np.zeros(4), np.zeros(4), kind)
    assert overflow == 1
    car_rows = np.flatnonzero(k // 10_000 == 0)
    assert sorted(pos[car_rows, 0].tolist()) == [1.0, 2.0]
    taxi_row = np.flatnonzero(k // 10_000 == 1)[0]
    assert pos[taxi_row, 0] == 4.0
    assert np.allclose(np.linalg.norm(rot, axis=1), 1.0)


def test_unused_slots_are_parked_out_of_sight():
    k = slots({"car": 3, "taxi": 0, "bus": 0, "moto": 0, "truck": 0})
    pos, _, _ = slot_transforms(k, np.zeros((1, 3)), np.zeros(1), np.zeros(1), np.zeros(1, int))
    assert (pos[:, 1] == PARKED).sum() == 2


def test_heading_turns_the_nose():
    k = slots({"car": 1, "taxi": 0, "bus": 0, "moto": 0, "truck": 0})
    _, rot, _ = slot_transforms(k, np.zeros((1, 3)), np.array([np.pi / 2]), np.zeros(1), np.zeros(1, int))
    x, y, z, w = rot[0]
    nose = np.array([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)])  # +x rotated
    assert np.allclose(nose, [0, 0, 1], atol=1e-6)  # heading pi/2 is south, +z


@pytest.mark.slow
def test_transforms_for_a_thousand_cars_are_fast():
    pool = {"car": 800, "taxi": 400, "bus": 60, "moto": 400, "truck": 200}
    k = slots(pool)
    rng = np.random.default_rng(0)
    xyz, yaw, pitch = rng.random((1000, 3)), rng.random(1000), rng.random(1000)
    kind = rng.integers(0, 5, 1000)
    t0 = time.perf_counter()
    for _ in range(20):
        slot_transforms(k, xyz, yaw, pitch, kind)
    ms = (time.perf_counter() - t0) / 20 * 1000
    print(f"slot transforms for 1000 cars in {len(k)} slots: {ms:.2f} ms")
    assert ms < 2.0
