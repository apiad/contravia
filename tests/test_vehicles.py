import time

import numpy as np
import pytest

from contravia.rules.fleet import KINDS
from contravia.rules.traffic import look_kinds, looks
from contravia.traffic import LENGTH
from contravia.vehicles import PARKED, SLOT, _shape, pool_sizes, slot_transforms, slots

KIND = look_kinds()
CAR, OTHER_CAR = np.flatnonzero(KIND == KINDS.index("car"))[[0, -1]]
TAXI = np.flatnonzero(KIND == KINDS.index("taxi"))[0]


@pytest.mark.slow
def test_one_look_of_every_kind_builds_of_its_length():
    for kind in KINDS:
        _, family, k, _ = next(look for look in looks() if look[0] == kind)
        for m in _shape(kind, family, k):
            span = np.ptp(m["positions"][:, 0])
            assert span == pytest.approx(LENGTH[kind], rel=0.3), (kind, span)
            assert len(m["indices"]) // 3 < 30_000, kind
            assert m["colors"].shape == m["positions"].shape


def test_every_kind_has_a_pool():
    pool = pool_sizes()
    assert {KINDS[KIND[i]] for i in pool} == set(KINDS)
    assert sum(pool.values()) < 2000


def test_cars_fill_their_look_then_their_kind_and_overflow_is_counted():
    k = slots({CAR: 2, OTHER_CAR: 1, TAXI: 1})
    xyz = np.array([[1.0, 0, 0], [2.0, 0, 0], [3.0, 0, 0], [4.0, 0, 0], [5.0, 0, 0]])
    look = np.array([CAR, CAR, CAR, CAR, TAXI])  # four cars of one look for its two slots and one of its kind's
    pos, rot, overflow = slot_transforms(k, xyz, np.zeros(5), np.zeros(5), look)
    assert overflow == 1
    assert sorted(pos[k // SLOT == CAR, 0].tolist()) == [1.0, 2.0]
    assert pos[k // SLOT == OTHER_CAR, 0].tolist() == [3.0]
    assert pos[k // SLOT == TAXI, 0].tolist() == [5.0]
    assert np.allclose(np.linalg.norm(rot, axis=1), 1.0)


def test_unused_slots_are_parked_out_of_sight():
    k = slots({CAR: 3})
    pos, _, _ = slot_transforms(k, np.zeros((1, 3)), np.zeros(1), np.zeros(1), np.array([CAR]))
    assert (pos[:, 1] == PARKED).sum() == 2


def test_heading_turns_the_nose():
    k = slots({CAR: 1})
    _, rot, _ = slot_transforms(k, np.zeros((1, 3)), np.array([np.pi / 2]), np.zeros(1), np.array([CAR]))
    x, y, z, w = rot[0]
    nose = np.array([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)])  # +x rotated
    assert np.allclose(nose, [0, 0, 1], atol=1e-6)  # heading pi/2 is south, +z


@pytest.mark.slow
def test_transforms_for_a_thousand_cars_are_fast():
    k = slots(pool_sizes(1000))
    rng = np.random.default_rng(0)
    xyz, yaw, pitch = rng.random((1000, 3)), rng.random(1000), rng.random(1000)
    look = rng.integers(0, len(KIND), 1000)
    t0 = time.perf_counter()
    for _ in range(20):
        slot_transforms(k, xyz, yaw, pitch, look)
    ms = (time.perf_counter() - t0) / 20 * 1000
    print(f"slot transforms for 1000 cars in {len(k)} slots: {ms:.2f} ms")
    assert ms < 2.0
