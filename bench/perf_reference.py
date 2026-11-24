"""Controlled real-game core references, separate from route/history benchmarks.

Each sample restores the same complete solver state. The contact case replays a
two-car impact; it is neither normal traffic nor a driving route. Private engine
hooks are deliberately confined here, pinned by package provenance and CPU tests.
"""

import argparse
import base64
import hashlib
import html
import importlib.metadata
import json
import math
import os
import signal
import subprocess
import sys
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SCENES = ("controlled-parked-free", "controlled-vehicle-contact")
DT = 1 / 60
WHEN = "2026-11-24 10:00"


def fingerprint(value):
    """Hash JSON state without losing floating point precision."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    """Hash bytes in bounded memory, including large island inputs."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def percentiles(values):
    """Linear p50/p95/p99; absent metrics remain unavailable."""
    if not values:
        return None
    a = sorted(values)
    if not all(math.isfinite(v) and v >= 0 for v in a):
        raise ValueError("non-finite or negative measurement")
    def at(p):
        k = (len(a) - 1) * p
        lo = int(k)
        return a[lo] + (a[min(lo + 1, len(a) - 1)] - a[lo]) * (k - lo)
    return dict(zip(("p50", "p95", "p99"), (at(.5), at(.95), at(.99))))


def summarize(samples):
    """Aggregate actual work, without summing overlapping engine timings."""
    if not samples:
        raise ValueError("empty samples")
    stepped = [s for s in samples if s["substeps"] > 0]
    steps = sum(s["substeps"] for s in stepped)
    return {
        "samples": len(samples),
        **{k: percentiles([s[k] for s in samples if s[k] is not None])
           for k in ("frame_wall_ms", "frame_cpu_ms", "physics_wall_ms", "gpu_total_ms", "rss_bytes")},
        "physics_ms_per_executed_substep": sum(s["physics_wall_ms"] for s in stepped) / steps if steps else None,
        "zero_step_calls": len(samples) - len(stepped),
        "zero_step_call_ms": percentiles([s["physics_wall_ms"] for s in samples if not s["substeps"]]),
        "substeps": dict(Counter(s["substeps"] for s in samples)),
        "bodies": dict(Counter(s["bodies"] for s in samples)),
        "contact_samples": sum(s["contacts"] > 0 for s in samples),
    }


def validate(run, expected):
    """Validate measured facts against the requested contract, not its summary."""
    if run["contract"] != expected:
        raise ValueError("requested size/preset/seed/scene/timestep contract differs")
    if expected["scene"] not in SCENES or run["traffic"] != "disabled-controlled":
        raise ValueError("controlled scene cannot claim normal traffic or driving")
    observed = run["observed"]
    if (observed["size"] != expected["size"] or observed["preset"] != expected["preset"]
            or observed["seed"] != expected["seed"] or observed["tune"]
            or observed["traffic_on"] or observed["time"] != WHEN or observed["weather"] != "clear"):
        raise ValueError("observed scene settings differ from requested settings")
    samples = run["samples"]
    if len(samples) != expected["frames"] or not samples:
        raise ValueError("missing or empty samples")
    if fingerprint(run["initial_state"]) != run["state_fingerprint"]:
        raise ValueError("initial state fingerprint differs")
    if fingerprint(run["visual_work"]) != run["work_fingerprint"]:
        raise ValueError("visual work fingerprint differs")
    contact = expected["scene"] == SCENES[1]
    for s in samples:
        if s["bodies"] != (2 if contact else 1) or s["waiting"]:
            raise ValueError("wrong body count or unready streaming")
        if s["contacts"] < 0 or not math.isfinite(s["impulse_ns"]) or s["impulse_ns"] < 0:
            raise ValueError("invalid contact measurement")
        if s["input_fingerprint"] != run["state_fingerprint"] or s["work_fingerprint"] != run["work_fingerprint"]:
            raise ValueError("sample state/work differs")
        if s["waiting"] or s["tune"] or s["size"] != expected["size"] or s["preset"] != expected["preset"]:
            raise ValueError("unready streaming, tuning or wrong rendered configuration")
        if s["bodies"] != 1 + int(contact) or type(s["substeps"]) is not int or s["substeps"] < 0:
            raise ValueError("invalid solver work")
        if s["substeps"] != s["observed_steps"]:
            raise ValueError("telemetry disagrees with actual solver steps")
        if not contact and (s["contacts"] or s["impulse_ns"]):
            raise ValueError("contact-free scene had a rigid-body contact")
        if s["substeps"] == 0 and (s["contacts"] or s["impulse_ns"]):
            raise ValueError("zero-step call reported stale contact")
        if expected["timestep"] == "fixed-physics" and s["substeps"] != 2:
            raise ValueError("fixed physics did not execute two substeps")
        if not math.isfinite(s["displacement_m"]):
            raise ValueError("non-finite vehicle state")
    if contact and not any(s["contacts"] > 0 and s["impulse_ns"] > 0 and s["displacement_m"] > 0 for s in samples):
        raise ValueError("contact scene has no solver contact/impulse/movement")
    summarize(samples)


def compare(left, right):
    """Only equivalent state, images' work and solver distributions may compare."""
    validate(left, left["contract"])
    validate(right, right["contract"])
    for k in ("contract", "state_fingerprint", "work_fingerprint", "data_fingerprint", "camera", "controls", "quality"):
        if left[k] != right[k]:
            raise ValueError(f"incomparable {k}")
    for k in ("substeps", "bodies"):
        if Counter(s[k] for s in left["samples"]) != Counter(s[k] for s in right["samples"]):
            raise ValueError(f"incomparable {k} distributions")
    # Real-time dt varies. Require identical resulting state as well as work;
    # rejection is preferable to silently comparing different simulations.
    if [s["result_fingerprint"] for s in left["samples"]] != [s["result_fingerprint"] for s in right["samples"]]:
        raise ValueError("incomparable resulting solver states")


class SolverProbe:
    """Observe actual solver ticks and active contact resolutions, including scalar paths."""

    def __init__(self, driven):
        self.driven = driven
        self.reset()

    def reset(self):
        self.steps = self.contacts = 0
        self.impulse = self.wall_ms = 0.0

    @contextmanager
    def installed(self):
        from manifoldx.vehicles import single
        f = self.driven.fleet
        old_step, old_apply, old_pair, old_single = f.step, f._apply, f._pair, single._apply

        def step(*args, **kwargs):
            result = old_step(*args, **kwargs)
            self.steps += 1
            self.impulse += float(f.impulse.sum())
            return result

        def apply(inv, lever, normal, depth):
            self.contacts += int((depth > 0).sum())
            return old_apply(inv, lever, normal, depth)

        def pair(*args, **kwargs):
            self.contacts += 1  # _bodies calls _pair only after a positive box overlap
            return old_pair(*args, **kwargs)

        def scalar(body, lever, normal, depth):
            self.contacts += int(depth > 0)
            return old_single(body, lever, normal, depth)

        f.step, f._apply, f._pair, single._apply = step, apply, pair, scalar
        try:
            yield self
        finally:
            f.step, f._apply, f._pair, single._apply = old_step, old_apply, old_pair, old_single


def solver_state(driven):
    """All mutable solver arrays, accumulator and interpolation endpoints."""
    f = driven.fleet
    return {**{k: getattr(f, k).tolist() for k in f.STATE}, "acc": f._acc,
            "before": [a.tolist() for a in f._before]}


def restore(driven, state):
    """Reset every solver row, including leftovers; no seed-only determinism."""
    import numpy as np
    f = driven.fleet
    for k in f.STATE:
        getattr(f, k)[...] = state[k]
    f._acc = state["acc"]
    f._before = tuple(np.asarray(a).copy() for a in state["before"])
    driven.substeps = 0
    driven.state = driven._read()


def provenance(args):
    """Require a pinned installed engine and force game imports from this checkout."""
    engine_path = Path(args.engine_path).resolve()
    sys.path.insert(1, str(engine_path.parent))
    import manifoldx

    import contravia
    if Path(manifoldx.__file__).resolve().parent != engine_path:
        raise ValueError("engine import did not resolve to --engine-path")
    if Path(contravia.__file__).resolve().parent != ROOT / "contravia":
        raise ValueError("game import did not resolve to this worktree")
    dist = importlib.metadata.distribution("manifold-gfx")
    direct = json.loads(dist.read_text("direct_url.json") or "{}")
    archive = json.loads(Path(args.source_manifest).read_text()) if getattr(args, "source_manifest", None) else None
    revision = archive["engine"]["revision"] if archive else direct.get("vcs_info", {}).get("commit_id")
    if revision != args.engine_revision or len(args.engine_revision) != 40:
        raise ValueError("engine revision differs from pinned --engine-revision (full Git SHA required)")
    sources = {str(p.relative_to(ROOT)): file_hash(p) for p in sorted((ROOT / "contravia").rglob("*.py"))}
    sources["bench/perf_reference.py"] = file_hash(__file__)
    def git(*cmd):
        return subprocess.check_output(["git", "-C", str(ROOT), *cmd], text=True).strip()
    if archive:
        verify_files(ROOT, archive["game"]["files"])
        verify_files(engine_path, archive["engine"]["files"])
        actual_engine = {str(p.relative_to(engine_path)) for p in engine_path.rglob("*")
                         if p.is_file() and "__pycache__" not in p.parts}
        if actual_engine != set(archive["engine"]["files"]) or not set(sources) <= set(archive["game"]["files"]):
            raise ValueError("source archive manifest does not cover the loaded runtime")
    return {"python": sys.version, "executable": sys.executable, "engine_path": str(engine_path),
            "engine_origin": "verified-source-archive" if archive else "installed-distribution",
            "engine_revision": revision, "environment_distribution_version": dist.version,
            "environment_distribution_direct_url": direct,
            "source_manifest": archive,
            "engine_record_sha256": fingerprint(dist.read_text("RECORD")),
            "engine_sources": {str(p.relative_to(engine_path)): file_hash(p)
                               for p in sorted(engine_path.rglob("*.py"))},
            "game_path": str(ROOT), "game_sha": archive["game"]["sha"] if archive else git("rev-parse", "HEAD"),
            "game_status": "verified archive" if archive else git("status", "--porcelain"), "game_sources": sources,
            "argv": sys.argv}


def verify_files(root, files):
    """Check actual source archive bytes, refusing missing or modified runtime files."""
    for name, expected in files.items():
        path = (root / name).resolve()
        if root not in path.parents or file_hash(path) != expected:
            raise ValueError(f"source archive differs: {name}")


def protect_data(data, output):
    """Redirect recomputable baked caches and refuse any write to the input island."""
    from contravia import baked
    original = baked.baked
    def cached(path, stamp, build):
        target = output / "cache" / (fingerprint(str(Path(path).resolve())) + ".pkl")
        target.parent.mkdir(exist_ok=True)
        return original(target, stamp, build)
    baked.baked = cached

    def audit(event, args):
        if event == "open":
            path, mode, flags = args
            writing = (mode and any(c in mode for c in "wax+")) or (flags & (os.O_WRONLY | os.O_RDWR))
            if writing and isinstance(path, (str, bytes, os.PathLike)):
                p = Path(os.fsdecode(path)).resolve()
                if p == data or data in p.parents:
                    raise PermissionError(f"reference data is read-only: {p}")
    sys.addaudithook(audit)


def data_manifest(data):
    """Fingerprint actual inputs, excluding recomputable caches and prior check outputs."""
    files = sorted(p for p in data.rglob("*") if p.is_file() and p.suffix in (".npz", ".src", ".npy"))
    if not (data / "index.npz").is_file() or len(list(data.glob("*.npz"))) < 17:
        raise ValueError("an existing indexed, complete island directory is required")
    return {str(p.relative_to(data)): file_hash(p) for p in files}


def work(game):
    """Stable drawn geometry identities and counts, rather than asynchronous completion counters."""
    sm = game.streamer
    return {"nodes": sorted(map(list, sm.drawn())), "detail": sorted(map(list, game.detail.drawn())),
            "triangles": sum(len(c.geo["indices"]) // 3 for n, c in sm._ready.items()
                             if c is not None and n in sm.drawn()),
            "people": {k: v for k, v in game.crowd.counts().items() if k != "figures"},
            "forest": game.forest.counts(), "grass": game.grass.counts(), "graveyard": game.graveyard.counts(),
            "vehicles": len(game.driven.fleet.position),
            "tau": sm.tau}


def waiting(game):
    """Reject missing selected nodes and every visible decoration still being built."""
    return (game.waiting() + len(game.streamer._selected - game.streamer.drawn())
            + sum(x.waiting() for x in (game.forest, game.grass, game.graveyard, game.crowd))
            + int(game.crowd.figures.filling))


def rss_bytes():
    """Current Linux resident pages, not process high-water memory."""
    return int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")


def worker(args, output):
    """Load, warm, capture and measure one isolated repetition of the real game."""
    import random

    import manifoldx as mx
    import numpy as np
    from PIL import Image

    from contravia.car import CarState, Controls, Driven
    from contravia.clock import Clock, parse_at
    from contravia.game import WALL_REACH, Game
    from contravia.geo import to_local
    from contravia.quality import PRESETS
    from contravia.walls import around
    from contravia.world import SUMMARY_VERSION, World, _summary_version

    start = time.monotonic()
    data = Path(args.data).resolve()
    os.environ["HAVANA_DATA"] = str(data)
    os.environ["CONTRAVIA_CONFIG"] = str(output / "config")
    os.environ["XDG_CACHE_HOME"] = str(output / "cache-home")
    protect_data(data, output)
    os.environ["CONTRAVIA_PERF_DATA"] = str(data)
    os.environ["CONTRAVIA_PERF_OUTPUT"] = str(output)
    manifest = data_manifest(data)
    if _summary_version(data) < SUMMARY_VERSION:
        raise ValueError("island index is stale; regeneration is forbidden in this runner")
    random.seed(args.seed)
    np.random.seed(args.seed)
    engine = mx.Engine("contravia-perf-reference", width=args.size[0], height=args.size[1], pipelined=False,
                       smooth_dt=False)
    game = None
    try:
        clock = Clock(at=parse_at(WHEN), warp=0)
        clock.pause()
        game = Game(engine, World.load(data), at=tuple(to_local(23.1446264653, -82.3799149664)),
                    clock=clock, with_traffic=False)
        game.set_quality(args.preset, PRESETS[args.preset], pinned=True)
        game.tune = False
        game.help_level = 5
        game._rng = np.random.default_rng(args.seed)
        game.crowd.reseed(args.seed)
        game.crowd.set_limit(0)
        game.force_weather({"state": "clear", "minutes": 60})
        controls = Controls(0.0, 0.0, 0.0, True)
        game.autopilot = lambda car: controls
        x, z, y = game.car.x, game.car.z, game.car.y
        # Pin the pose after the real road snap. Camera does not chase numerical drift.
        game.driven.place(x, z, 0.0, y)
        game.car = game.driven.state
        camera = ((x - 9, y + 5, z + 10), (x + 3, y + 1, z))
        game.camera_override = lambda: camera
        engine.set_fixed_timestep(DT)
        original_dt = engine._compute_dt
        def pinned_visual_time():
            dt = original_dt()
            engine.elapsed = 0.0
            return dt
        engine._compute_dt = pinned_visual_time
        engine._ensure_offscreen(1)
        loaded = time.monotonic() - start
        warm_start, stable, previous = time.monotonic(), 0, None
        while stable < 30:
            if time.monotonic() - warm_start > args.warmup_timeout:
                raise TimeoutError("streaming did not become ready and stable")
            engine._draw_frame()
            engine._render_canvas.draw()
            current = fingerprint(work(game))
            stable = stable + 1 if not waiting(game) and current == previous else 0
            previous = current
        warm_seconds = time.monotonic() - warm_start
        # Streaming completion takes a variable number of frames. Never use the
        # solver state reached by that variable warmup as a reference input.
        walls = around(game.walls.near(x, z, y, wait=True), x, z, WALL_REACH + 16 * DT)
        game.driven = Driven(game.driven.spec, CarState(x, z, 0.0, 0.0, y=y), game.surface)
        for _ in range(120):
            game.driven.step(controls, DT, walls, game.strengths())
        game.car = game.driven.state
        contact = args.scene == SCENES[1]
        f = game.driven.fleet
        second_handles = None
        if contact:
            from manifoldx.modeling.assets.car import spawn_car
            f.add(game.driven.spec)
            f.place(1, (x + float(f.p.size[0, 0]) - .05, y, z), 0, game.driven.ground)
            # Settled player into the second car: controlled impact, reset per sample.
            game.driven.launch(8.0)
            controls = Controls(0.0, 0.0, 0.0, False)
            second_handles = spawn_car(engine, game.model)
        f._acc = 0.0
        f._before = (f.position.copy(), f.rotation.copy())
        initial = solver_state(game.driven)
        initial_hash = fingerprint(initial)
        probe = SolverProbe(game.driven)
        original_step = game.driven.step

        def measured(c, dt, walls, strengths=None, others=None):
            probe.reset()
            t = time.perf_counter()
            state = original_step(c, dt, walls, strengths,
                                  np.zeros((1, 4)) if contact else others)
            probe.wall_ms = (time.perf_counter() - t) * 1000
            if second_handles:
                from manifoldx.modeling.assets.car import place_car
                at, rotation = f.model_transform(1)
                place_car(second_handles, game.model, at=tuple(at), rotation=tuple(rotation),
                          travel=f.travel(1).tolist())
            return state

        game.driven.step = measured
        visual = work(game)
        visual_hash = fingerprint(visual)
        samples = []
        with probe.installed():
            # A real capture of this scenario, outside the measured region.
            restore(game.driven, initial)
            game.car = game.driven.state
            engine._draw_frame()
            image = np.asarray(engine._render_canvas.draw())
            Image.fromarray(image).convert("RGB").save(output / "scene.png")
            if args.timestep == "realtime-frame":
                engine.use_wall_clock()
                engine._last_time = engine._start_time = time.perf_counter_ns()
            for _ in range(args.frames):
                restore(game.driven, initial)
                game.car = game.driven.state
                input_hash = fingerprint(solver_state(game.driven))
                t, cpu = time.perf_counter(), time.process_time()
                if args.timestep == "fixed-physics":
                    measured(controls, DT, walls, game.strengths())
                else:
                    if not engine._draw_frame():
                        raise RuntimeError("engine stopped before all samples")
                    engine._render_canvas.draw()
                wall_ms, cpu_ms = (time.perf_counter() - t) * 1000, (time.process_time() - cpu) * 1000
                telemetry = game.driven.state.telemetry
                gpu = engine.gpu_timings() if args.timestep == "realtime-frame" else None
                samples.append({"frame_wall_ms": wall_ms if args.timestep == "realtime-frame" else None,
                                "frame_cpu_ms": cpu_ms if args.timestep == "realtime-frame" else None,
                                "physics_wall_ms": probe.wall_ms,
                                "dt_s": DT if args.timestep == "fixed-physics" else game._dt,
                                "gpu_total_ms": gpu.get("total") if gpu else None,
                                "gpu_pass_ms": gpu, "rss_bytes": rss_bytes(),
                                "substeps": telemetry["substeps"], "bodies": telemetry["bodies"],
                                "observed_steps": probe.steps, "contacts": probe.contacts,
                                "impulse_ns": probe.impulse, "input_fingerprint": input_hash,
                                "result_fingerprint": fingerprint(solver_state(game.driven)),
                                "work_fingerprint": fingerprint(work(game)), "waiting": waiting(game),
                                "tune": game.tune, "size": [engine.w, engine.h], "preset": game.quality_name,
                                "displacement_m": float(np.linalg.norm(f.position[0] - np.asarray(initial["position"])[0])),
                                "streamer": game.streamer.stats(),
                                "frame_split_ms": {k: v * 1000 for k, v in engine.frame_split.items()},
                                "system_split_ms": {k: v * 1000 for k, v in engine.system_split.items()}})
        observed = {"size": [engine.w, engine.h], "preset": game.quality_name, "seed": args.seed,
                    "tune": game.tune, "traffic_on": game.layer is not None,
                    "time": game.clock.now().strftime("%Y-%m-%d %H:%M"), "weather": game.weather_state}
        contract = {k: getattr(args, k) for k in ("scene", "size", "preset", "seed", "frames", "timestep")}
        return {"contract": contract, "observed": observed, "traffic": "disabled-controlled",
                "initial_state": initial, "state_fingerprint": initial_hash,
                "visual_work": visual, "work_fingerprint": visual_hash,
                "data_manifest": manifest, "data_fingerprint": fingerprint(manifest),
                "camera": camera, "controls": vars(controls), "quality": engine.state.quality,
                "adapter": dict(engine._adapter.info), "load_seconds": loaded,
                "screenshot_sha256": file_hash(output / "scene.png"),
                "warmup_seconds": warm_seconds, "samples": samples,
                "measurement": "serial offscreen, no vsync; reset solver state before each sample; probe overhead included"}
    finally:
        if game is not None:
            game._shutdown()
        engine.close()


def contract_for(args):
    """The requested comparison identity, independent of measured facts."""
    return {k: getattr(args, k) for k in ("scene", "size", "preset", "seed", "frames", "timestep")}

def report(output, runs):
    """A self-contained review page with actual captured pixels and raw summaries."""
    cards = []
    for i, run in enumerate(runs):
        png = base64.b64encode((output / str(i) / "scene.png").read_bytes()).decode()
        cards.append(f'<h2>Repeat {i + 1}</h2><img alt="Captured controlled scene" src="data:image/png;base64,{png}">'
                     f'<pre>{html.escape(json.dumps(run["summary"], indent=2))}</pre>')
    page = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
    page += '<title>Contravía controlled performance reference</title><style>body{max-width:1000px;margin:2em auto;'
    page += 'padding:0 1em;font:16px system-ui;background:#151515;color:#eee}img{width:100%}pre{overflow:auto}</style>'
    page += '<h1>Controlled performance reference</h1><p>Not normal traffic, route driving or display FPS. '
    page += 'Fixed physics and real-time offscreen frames are separate measurements. GPU unavailable is null. '
    page += 'Solver instrumentation is included in cost; overlapping splits are never added.</p>'
    page += '<pre>' + html.escape(json.dumps(runs[0]["contract"], indent=2)) + '</pre>' + ''.join(cards) + '</html>'
    (output / "report.html").write_text(page)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--core", choices=("python", "rust"), required=True)
    parser.add_argument("--engine-path", required=True, help="installed manifoldx package directory")
    parser.add_argument("--engine-revision", required=True, help="installed engine's full direct_url Git SHA")
    parser.add_argument("--source-manifest", type=Path, help="hash manifest for explicit game/engine source archives")
    parser.add_argument("--size", required=True)
    parser.add_argument("--preset", choices=("bajo", "medio", "alto", "ultra", "cine"), required=True)
    parser.add_argument("--scene", choices=SCENES, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--repeats", type=int, required=True)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--timestep", choices=("fixed-physics", "realtime-frame"), required=True)
    parser.add_argument("--warmup-timeout", type=float, default=120)
    parser.add_argument("--timeout", type=float, default=240, help="hard timeout per isolated repetition")
    parser.add_argument("--compare", type=Path, help="previous repeat's raw.json; reject unequal work/state")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.core != "python":
        parser.error("CORE=rust is unavailable: no integrated native core; refusing a Python fallback")
    if min(args.repeats, args.frames, args.timeout, args.warmup_timeout) <= 0:
        parser.error("frames, repeats and timeouts must be positive")
    try:
        args.size = [int(v) for v in args.size.split("x")]
        if len(args.size) != 2 or min(args.size) <= 0:
            raise ValueError()
    except ValueError:
        parser.error("size must be positive WIDTHxHEIGHT")
    output = Path(args.output).resolve()
    data = Path(args.data).resolve()
    if output == data or data in output.parents or output == ROOT or ROOT / "contravia" in output.parents:
        parser.error("output must be separate from inputs and source")
    output.mkdir(parents=True, exist_ok=False)
    try:
        prov = provenance(args)
        if args.worker:
            run = worker(args, output)
            run["provenance"] = prov
            validate(run, contract_for(args))
            run["summary"] = summarize(run["samples"])
            (output / "raw.json").write_text(json.dumps(run, indent=2, allow_nan=False))
            return
        runs = []
        raw_argv = list(argv if argv is not None else sys.argv[1:])
        for i in range(args.repeats):
            child_args = raw_argv.copy()
            child_args[child_args.index("--output") + 1] = str(output / str(i))
            with (output / f"{i}.log").open("w") as log:
                child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), *child_args, "--worker"],
                                         start_new_session=True, stdout=log, stderr=subprocess.STDOUT)
                try:
                    code = child.wait(timeout=args.timeout)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)  # include spawned builders; no orphan processes
                    child.wait()
                    raise
                if code:
                    raise RuntimeError(f"repeat {i} failed with status {code}; inspect {output / f'{i}.log'}")
            run = json.loads((output / str(i) / "raw.json").read_text())
            validate(run, contract_for(args))
            if runs:
                compare(runs[0], run)
            if args.compare:
                compare(json.loads(args.compare.read_text()), run)
            runs.append(run)
            print(f"{i + 1}/{args.repeats} repetitions validated", flush=True)
        report(output, runs)
        (output / "summary.json").write_text(json.dumps([r["summary"] for r in runs], indent=2))
    except Exception as error:
        (output / "failure.json").write_text(json.dumps({"error": str(error), "valid_reference": False}))
        raise


if __name__ == "__mp_main__" and "CONTRAVIA_PERF_DATA" in os.environ:
    protect_data(Path(os.environ["CONTRAVIA_PERF_DATA"]), Path(os.environ["CONTRAVIA_PERF_OUTPUT"]))

if __name__ == "__main__":
    main()
