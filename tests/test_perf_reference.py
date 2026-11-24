"""CPU proof of the reference probe, aggregation and rejection gates."""

import importlib.util
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from manifoldx.vehicles import FlatGround

from contravia.car import CarState, Controls, Driven
from contravia.rules import fleet, handling

SPEC = importlib.util.spec_from_file_location("perf_reference", Path(__file__).parents[1] / "bench/perf_reference.py")
perf = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(perf)
NONE = np.zeros((0, 2, 2))
IDLE = Controls(0, 0, 0, True)


def driven():
    return Driven(handling.spec_for(fleet.moderno(), "moderno"), CarState(0, 0, 0, 0), FlatGround())


def test_probe_checks_every_real_substep_and_does_not_reuse_stale_impulses():
    d = driven()
    probe = perf.SolverProbe(d)
    with probe.installed():
        s = d.step(IDLE, 1 / 240, NONE)
        assert s.telemetry["substeps"] == probe.steps == probe.contacts == probe.impulse == 0
        d.step(IDLE, 1 / 240, NONE)
        assert probe.steps == 1
        probe.reset()
        d.step(IDLE, 1 / 30, NONE)
        assert probe.steps == 4 and probe.contacts == probe.impulse == 0


def impact(d):
    d.fleet.add(d.spec)
    d.fleet.place(1, (float(d.fleet.p.size[0, 0]) - .05, 0, 0), 0, d.ground)
    d.launch(8)


def test_two_vehicle_contact_is_real_and_the_complete_state_replays():
    d = driven()
    impact(d)
    initial = perf.solver_state(d)
    probe = perf.SolverProbe(d)
    results = []
    with probe.installed():
        for _ in range(2):
            perf.restore(d, initial)
            assert perf.solver_state(d) == initial
            probe.reset()
            s = d.step(Controls(0, 0, 0, False), 1 / 60, NONE, others=np.zeros((1, 4)))
            assert s.telemetry["bodies"] == 2 and s.telemetry["substeps"] == probe.steps == 2
            assert probe.contacts > 0 and probe.impulse > 100
            assert probe.pair_contacts > 0 and probe.pair_impulse > 100
            results.append(perf.solver_state(d))
    assert results[0] == results[1] != initial


def test_scalar_wall_contacts_are_observed_and_disappear_when_wall_is_removed():
    d = driven()
    d.launch(8)
    initial = perf.solver_state(d)
    wall = np.array([[[1.0, -10.0], [1.0, 10.0]]])
    probe = perf.SolverProbe(d)
    with probe.installed():
        d.step(IDLE, 1 / 60, wall)
        assert probe.contacts > 0 and probe.impulse > 0
        perf.restore(d, initial)
        probe.reset()
        d.step(IDLE, 1 / 60, NONE)
        assert probe.contacts == probe.impulse == 0


def valid_run(contact=False):
    """A valid fixture whose work counters come from a real CPU solver run."""
    d = driven()
    if contact:
        impact(d)
    initial = perf.solver_state(d)
    probe = perf.SolverProbe(d)
    with probe.installed():
        d.step(IDLE, 1 / 60, NONE, others=np.zeros((1, 4)) if contact else None)
    state = perf.fingerprint(initial)
    visual = {"nodes": [[0, 0, 0]], "vehicles": 1 + int(contact)}
    work = perf.fingerprint(visual)
    contract = {"scene": perf.SCENES[int(contact)], "size": [1920, 1080], "preset": "alto", "seed": 7, "frames": 1,
                    "timestep": "fixed-physics"}
    return {"contract": contract, "observed": {"size": [1920, 1080], "preset": "alto", "seed": 7, "tune": False,
                "traffic_on": False, "time": perf.WHEN, "weather": "clear"}, "traffic": "disabled-controlled",
                "initial_state": initial, "state_fingerprint": state, "visual_work": visual, "work_fingerprint": work,
                "data_fingerprint": "fixture", "camera": [[0, 5, 10], [0, 0, 0]], "controls": vars(IDLE), "quality": "alto",
                "samples": [{"substeps": 2, "observed_steps": probe.steps, "bodies": 1 + int(contact),
                "contacts": probe.contacts, "impulse_ns": probe.impulse,
                "pair_contacts": probe.pair_contacts, "pair_impulse_ns": probe.pair_impulse,
                "displacement_m": float(np.linalg.norm(d.fleet.position[0] - np.array(initial["position"])[0])), "waiting": 0,
                "tune": False, "size": [1920, 1080], "preset": "alto",
                "input_fingerprint": state, "result_fingerprint": perf.fingerprint(perf.solver_state(d)),
                "work_fingerprint": work, "physics_wall_ms": 3.0, "frame_cpu_ms": None, "frame_wall_ms": None,
                "gpu_total_ms": None, "rss_bytes": 1024}]}


def test_valid_fixture_passes_before_mutations():
    r = valid_run()
    perf.validate(r, r["contract"])
    perf.compare(r, deepcopy(r))
    summary = perf.summarize(r["samples"])
    assert summary["physics_ms_per_executed_substep"] == 1.5
    assert summary["gpu_total_ms"] is None
    assert perf.percentiles([1, 2, 3, 4, 5])["p50"] == 3


@pytest.mark.parametrize("mutation", ["empty", "size", "preset", "seed", "state", "work", "steps",
                                    "bodies", "contact", "unready", "normal", "nan", "missing"])
def test_reject_mutated_measurements(mutation):
    r = valid_run()
    expected = deepcopy(r["contract"])
    perf.validate(r, expected)
    if mutation == "empty":
        r["samples"] = []
    elif mutation in ("size", "preset", "seed"):
        r["observed"][mutation] = {"size": [1280, 720], "preset": "bajo", "seed": 8}[mutation]
    elif mutation == "state":
        r["initial_state"]["position"][0][0] += 1
    elif mutation == "work":
        r["visual_work"]["vehicles"] = 2
    elif mutation == "steps":
        r["samples"][0]["substeps"] = 0
    elif mutation == "bodies":
        r["samples"][0]["bodies"] = 2
    elif mutation == "contact":
        r["samples"][0]["contacts"] = 1
    elif mutation == "unready":
        r["samples"][0]["waiting"] = 1
    elif mutation == "normal":
        r["traffic"] = "normal"
    elif mutation == "nan":
        r["samples"][0]["physics_wall_ms"] = float("nan")
    elif mutation == "missing":
        del r["samples"][0]["physics_wall_ms"]
    with pytest.raises((ValueError, KeyError)):
        perf.validate(r, expected)


def test_false_contact_and_motion_claims_fail_after_valid_contact_fixture_passes():
    r = valid_run(contact=True)
    perf.validate(r, r["contract"])
    for field in ("contacts", "impulse_ns", "displacement_m", "pair_contacts", "pair_impulse_ns"):
        bad = deepcopy(r)
        bad["samples"][0][field] = 0
        with pytest.raises(ValueError, match="no solver contact"):
            perf.validate(bad, bad["contract"])


def test_real_time_work_mismatch_is_rejected_even_with_the_same_seed():
    a = valid_run()
    a["contract"]["timestep"] = "realtime-frame"
    b = deepcopy(a)
    b["samples"][0].update(substeps=1, observed_steps=1)
    perf.validate(b, b["contract"])
    with pytest.raises(ValueError, match="substeps distributions"):
        perf.compare(a, b)


def test_zero_substep_cost_has_no_division_or_invented_physics_work():
    s = valid_run()["samples"][0]
    s["substeps"] = 0
    summary = perf.summarize([s])
    assert summary["physics_ms_per_executed_substep"] is None
    assert summary["zero_step_calls"] == 1 and summary["zero_step_call_ms"]["p50"] == 3
    with pytest.raises(ValueError, match="empty"):
        perf.summarize([])


def test_rust_refusal_happens_before_engine_import_or_output_creation(tmp_path):
    output = tmp_path / "out"
    result = subprocess.run([sys.executable, str(Path(perf.__file__)), "--core", "rust", "--data", "absent",
                             "--output", str(output), "--engine-path", "absent", "--engine-revision", "absent",
                             "--size", "1920x1080", "--preset", "alto", "--scene", perf.SCENES[0],
                             "--seed", "7", "--repeats", "1", "--timestep", "fixed-physics"],
                            check=False, capture_output=True, text=True)
    assert result.returncode != 0 and "refusing a Python fallback" in result.stderr
    assert not output.exists()


def test_wrong_engine_revision_fails_before_gpu_start():
    import manifoldx
    args = SimpleNamespace(engine_path=Path(manifoldx.__file__).parent, engine_revision="0" * 40)
    with pytest.raises(ValueError, match="revision differs"):
        perf.provenance(args)


def test_data_guard_blocks_real_writes_and_redirects_baked_cache(tmp_path):
    data, out = tmp_path / "island", tmp_path / "out"
    data.mkdir()
    out.mkdir()
    original = data / "source.npz"
    original.write_bytes(b"unchanged island")
    script = f'''
import importlib.util
from pathlib import Path
s=importlib.util.spec_from_file_location("perf", {str(perf.__file__)!r})
m=importlib.util.module_from_spec(s)
s.loader.exec_module(m)
data, out = Path({str(data)!r}), Path({str(out)!r})
m.protect_data(data, out)
from contravia.baked import baked
assert baked(data / "new.pkl", "key", lambda: [123]) == [123]
try:
    (data / "source.npz").write_bytes(b"corrupted")
except PermissionError:
    pass
else:
    raise AssertionError("input write was allowed")
'''
    subprocess.run([sys.executable, "-c", script], check=True)
    assert original.read_bytes() == b"unchanged island"
    assert not (data / "new.pkl").exists()
    assert len(list((out / "cache").glob("*.pkl"))) == 1


def test_source_archive_hashes_accept_real_bytes_and_reject_a_same_size_mutation(tmp_path):
    path = tmp_path / "engine.py"
    path.write_bytes(b"version=1")
    manifest = {"engine.py": perf.file_hash(path)}
    perf.verify_files(tmp_path, manifest)
    path.write_bytes(b"version=2")
    with pytest.raises(ValueError, match="source archive differs"):
        perf.verify_files(tmp_path, manifest)


def test_gpu_percentiles_count_new_readbacks_once_even_when_the_values_are_identical():
    timer = SimpleNamespace(last={"total": 2.0})
    engine = SimpleNamespace(_gpu_timer_obj=timer, gpu_timings=lambda: dict(timer.last))
    previous, sample = perf.fresh_gpu_sample(engine, timer.last)
    assert sample is None  # the warmup's completed query is not a measured GPU sample
    timer.last = {"total": 2.0}  # a genuinely new readback can have identical values
    previous, sample = perf.fresh_gpu_sample(engine, previous)
    assert sample == {"total": 2.0}
    previous, sample = perf.fresh_gpu_sample(engine, previous)
    assert sample is None  # repeat of the same readback must not inflate sample count
    engine._gpu_timer_obj = None
    assert perf.fresh_gpu_sample(engine, previous) == (None, None)
