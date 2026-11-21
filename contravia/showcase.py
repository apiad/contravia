"""contravia bench --showcase: three fixed views of La Habana, timed at one graphics preset, and a page to compare runs.

The scenes are in bench/showcase.json: the Malecón at sunset (the sea and a low sun), Obispo at noon (a narrow street
seen from a pedestrian's eye) and the hill over the Castillo del Príncipe (distance and fog), each at a fixed date and
time, so every run sees the same light. Each scene waits until the city around its first view is drawn, sets its
clock again, then moves its camera along a short path for `seconds` while it records
the frame time and, where manifoldx can measure it, the GPU time of each pass; it ends with a JPEG of its last view.
One run appends one line to bench/history.jsonl with the preset, the knobs changed from it and every scene's
numbers. `contravia bench --compare` writes one self-contained page with every run's numbers and shots side by
side, by scene.
"""

from __future__ import annotations

import base64
import html
import json
import platform
import time
from pathlib import Path

import numpy as np

from contravia.geo import to_local

SCENES = Path("bench/showcase.json")
SHOTS = Path("bench/out/showcase")  # gitignored, as bench/out is
PAGE = Path("bench/out/showcase.html")
SHOT_WIDTH = 1280  # pixels across a scene's JPEG
THUMB_WIDTH = 480  # pixels across a shot on the comparison page


def load(path: Path = SCENES) -> dict:
    """The scenes by name, without the file's note."""
    return {k: v for k, v in json.loads(Path(path).read_text()).items() if k != "note"}


def path_at(scene: dict, t: float, seconds: float) -> tuple:
    """The (eye, target) offsets of scene's camera t seconds into its run of `seconds`, spread evenly over its path."""
    pts = np.asarray(scene["path"], float)
    u = min(max(t / seconds, 0.0), 1.0) * (len(pts) - 1)
    k = min(int(u), len(pts) - 2)
    p = pts[k] + (pts[k + 1] - pts[k]) * (u - k)
    return tuple(float(v) for v in p[:3]), tuple(float(v) for v in p[3:])


def run(engine, game, scenes: dict, seconds: float = 30.0, wait: float = 300.0, shots: Path = SHOTS,
        tag: str = "run") -> dict:
    """Run each scene and return its numbers by name; its shot is saved as shots/<tag>-<name>.jpg."""
    from PIL import Image

    from contravia.clock import parse_at

    shots = Path(shots)
    shots.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, sc in scenes.items():
        x, z = to_local(*sc["at"])
        ground = float(game.terrain.height(x, z))
        game.clock.set(parse_at(sc["when"]))
        game.force_weather({"state": sc["weather"], "minutes": 600})
        game.teleport(x, z)
        start = time.monotonic()

        def view(t, sc=sc, x=x, z=z, ground=ground):
            (ex, ey, ez), (tx, ty, tz) = path_at(sc, t, seconds)
            return (x + ex, ground + ey, z + ez), (x + tx, ground + ty, z + tz)

        game.camera_override = lambda: view(0.0)
        engine.render_frame(supersample=1)
        while game.waiting() and time.monotonic() - start < wait:
            engine.render_frame(supersample=1)
        drawn = time.monotonic() - start
        game.clock.set(parse_at(sc["when"]))  # again: a slow preset waits longer, and its sun must not move on
        clock = game.clock.now().isoformat()
        ms, gpu, t0 = [], {}, time.monotonic()
        while not ms or time.monotonic() - t0 < seconds:
            game.camera_override = lambda t=time.monotonic() - t0: view(t)
            img = engine.render_frame(supersample=1)
            ms.append(game.frame_ms)
            timings = engine.gpu_timings() if hasattr(engine, "gpu_timings") else None
            for k, v in (timings or {}).items():
                gpu.setdefault(k, []).append(v)
        shot = Image.fromarray(np.asarray(img)).convert("RGB")
        if shot.width > SHOT_WIDTH:
            shot = shot.resize((SHOT_WIDTH, round(shot.height * SHOT_WIDTH / shot.width)))
        path = shots / f"{tag}-{name}.jpg"
        shot.save(path, quality=85)
        f = np.asarray(ms, float)
        out[name] = {"p50_ms": round(float(np.percentile(f, 50)), 2), "p95_ms": round(float(np.percentile(f, 95)), 2),
                     "frames": len(ms), "seconds_to_drawn": round(drawn, 1),
                     "gpu_ms": {k: round(float(np.median(v)), 3) for k, v in gpu.items()} or None,
                     "clock": clock, "shot": str(path)}
    game.camera_override = None
    return out


def main(data: str, preset: str = "alto", knobs: dict | None = None, size=(1920, 1080), seconds: float = 30.0) -> dict:
    """Run the scenes at a preset, or at Alto with some knobs changed, and append the run to the bench history."""
    import manifoldx as mx

    from contravia import bench
    from contravia.clock import Clock, parse_at
    from contravia.game import Game
    from contravia.quality import PRESETS
    from contravia.world import World

    q = PRESETS[preset].changed(**knobs) if knobs else PRESETS[preset]
    name = "personalizado" if knobs else preset
    engine = mx.Engine("contravia-showcase", width=size[0], height=size[1])
    game = Game(engine, World.load(data), clock=Clock(at=parse_at("Tue 08:00")), with_traffic=True)
    game.atmosphere.prepare(wait=True)
    game.set_quality(name, q, pinned=True)
    game.tune = False  # a fixed tau, so runs compare
    machine = bench._machine(engine)
    tag = f"{time.strftime('%Y%m%d-%H%M%S')}-{platform.node()}-{name}"
    try:
        scenes = run(engine, game, load(), seconds=seconds, tag=tag)
    finally:
        game._shutdown()
    result = {"showcase": True, "preset": preset, "knobs": q.diff(PRESETS[preset]), "scenes": scenes, **machine}
    bench.HISTORY.parent.mkdir(exist_ok=True)
    with bench.HISTORY.open("a") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def _thumb(path: str) -> str:
    """A shot as an inline JPEG THUMB_WIDTH pixels across, or "" when the file is gone."""
    from io import BytesIO

    from PIL import Image

    try:
        im = Image.open(path).convert("RGB")
    except OSError:
        return ""
    if im.width > THUMB_WIDTH:
        im = im.resize((THUMB_WIDTH, round(im.height * THUMB_WIDTH / im.width)))
    buf = BytesIO()
    im.save(buf, "JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


_STYLE = """body{margin:0;background:#11100e;color:#dcd9cf;font:14px system-ui,sans-serif}
main{max-width:1100px;margin:0 auto;padding:20px 16px 60px}h1{font-size:20px}h2{font-size:16px;margin-top:36px}
.runs{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:14px}
.run{background:#191814;border:1px solid #2a2721;border-radius:8px;padding:10px}
.run img{width:100%;height:auto;border-radius:4px;display:block}.k{color:#8a867b;font-size:12px}
.n{font-family:ui-monospace,monospace;font-size:12.5px;margin-top:6px;overflow-wrap:anywhere}"""


def compare(history: Path | None = None, out: Path = PAGE) -> Path:
    """Write one page with every showcase run of the history: per scene, a card per run with its shot and numbers."""
    from contravia import bench

    history = Path(history or bench.HISTORY)
    runs = [r for r in (json.loads(line) for line in history.read_text().splitlines() if line.strip())
            if r.get("showcase")]
    names = []
    for r in runs:
        names += [n for n in r["scenes"] if n not in names]
    parts = [f"<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport "
             f"content='width=device-width,initial-scale=1'><title>Contravía showcase bench</title>"
             f"<style>{_STYLE}</style></head><body><main><h1>Contravía showcase bench</h1>"
             f"<p class=k>{len(runs)} runs from {html.escape(str(history))}</p>"]
    for name in names:
        parts.append(f"<h2>{html.escape(name)}</h2><div class=runs>")
        for r in runs:
            s = r["scenes"].get(name)
            if not s:
                continue
            gpu = ", ".join(f"{k} {v} ms" for k, v in (s.get("gpu_ms") or {}).items()) or "no GPU timings"
            knobs = ", ".join(f"{k}={v}" for k, v in r.get("knobs", {}).items())
            src = _thumb(s.get("shot", ""))
            parts.append(
                "<div class=run>" + (f"<img alt='{html.escape(name)}' src='{src}'>" if src else "")
                + f"<div class=n>{html.escape(r['preset'])}{' · ' + html.escape(knobs) if knobs else ''}</div>"
                + f"<div class=k>{html.escape(str(r.get('host')))} · {html.escape(str(r.get('gpu')))} · "
                + f"{'x'.join(map(str, r.get('size', [])))}</div>"
                + f"<div class=n>p50 {s['p50_ms']} ms · p95 {s['p95_ms']} ms · {s['frames']} frames</div>"
                + f"<div class=n>{html.escape(gpu)} · VRAM {r.get('gpu_memory_mb') or 'n/a'} MB</div></div>")
        parts.append("</div>")
    parts.append("</main></body></html>")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(parts))
    return out
