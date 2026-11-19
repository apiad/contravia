"""The brand as the GUI draws it: the palette, the fonts, the icons, the poster's panels and a few shapes.

Everything here comes from `assets/` (written by assets/build.py) and `branding/poster/panels/`, found next to
the package in the checkout. The shapes are SVG markup drawn by manifoldx's `Image`, because the GUI's panels
are flat rectangles: the slanted yellow stripe of a chosen option with its red shadow, the cream ring of the
minimap, the dark fade behind the menu's column and the bars of the rev counter.
"""

import base64
import json
from functools import cache
from importlib.resources import files
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
PANELS = ROOT / "branding" / "poster" / "panels"
PALETTE = json.loads((ASSETS / "palette.json").read_text())
YELLOW, RED, INK, PAPER, GREY = (PALETTE[k] for k in ("yellow", "red", "ink", "paper", "grey"))
TITLE = str(ASSETS / "fonts" / "NotoSans-CondensedBlackItalic.ttf")  # titles, options, big numbers
BOLD = str(ASSETS / "fonts" / "NotoSans-CondensedBold.ttf")  # labels and small lines
# The brand fonts have no arrows: the key hints draw theirs with the engine's own font.
KEYS = str(files("manifoldx.viz.assets").joinpath("DejaVuSansMono.ttf"))
DESIGN_H = 720  # the GUI is laid out for a window this many pixels high and scaled with it


def icon(name: str) -> str:
    """The path of an icon in assets/icons, such as 'sol' or 'reloj'."""
    return str(ASSETS / "icons" / f"{name}.svg")


def logo() -> str:
    """The path of the full logo: the sign and the word."""
    return str(ASSETS / "logo" / "contravia.svg")


def panel_names() -> list[str]:
    """The poster's panels, by name, in a fixed order."""
    return sorted(p.stem for p in PANELS.glob("*.jpg"))


@cache
def panel(name: str) -> str:
    """A poster panel as SVG markup that holds the JPEG, so an Image can draw it; built once per panel."""
    from PIL import Image as Pil

    path = PANELS / f"{name}.jpg"
    w, h = Pil.open(path).size
    data = base64.b64encode(path.read_bytes()).decode()
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">'
            f'<image width="{w}" height="{h}" href="data:image/jpeg;base64,{data}"/></svg>')


def panel_aspect(name: str) -> float:
    """Width over height of a poster panel."""
    from PIL import Image as Pil

    w, h = Pil.open(PANELS / f"{name}.jpg").size
    return w / h


def cover_uv(image_aspect: float, box_aspect: float, zoom: float = 1.0) -> tuple[float, float, float, float]:
    """The part of an image (u0, v0, u1, v1) that fills a box of another aspect without bars, centred, and
    `zoom` times closer."""
    if image_aspect > box_aspect:
        w, h = box_aspect / image_aspect, 1.0
    else:
        w, h = 1.0, image_aspect / box_aspect
    w, h = w / zoom, h / zoom
    return (0.5 - w / 2, 0.5 - h / 2, 0.5 + w / 2, 0.5 + h / 2)


def stripe(w: int, h: int, slant: int = 12, shadow: int = 6) -> str:
    """The chosen option's stripe: a yellow parallelogram leaning right, with the logo's red shadow below
    and to its right."""
    s, d = slant, shadow
    poly = lambda dx, dy: " ".join(f"{x + dx},{y + dy}" for x, y in ((s, 0), (w - d, 0), (w - d - s, h - d),
                                                                         (0, h - d)))
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">'
            f'<polygon points="{poly(d, d)}" fill="{RED}"/><polygon points="{poly(0, 0)}" fill="{YELLOW}"/></svg>')


def ring(size: int, width: int = 6) -> str:
    """A cream ring with an ink edge, for the minimap's rim."""
    r = size / 2
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}">'
            f'<circle cx="{r}" cy="{r}" r="{r - width / 2 - 1}" fill="none" stroke="{INK}" stroke-width="{width + 2}"/>'
            f'<circle cx="{r}" cy="{r}" r="{r - width / 2 - 1}" fill="none" stroke="{PAPER}" stroke-width="{width}"/>'
            f'</svg>')


def badge(letter_path: str, size: int = 26) -> str:
    """A black disc with a cream rim and a yellow mark inside, as the minimap's north badge."""
    r = size / 2
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}">'
            f'<circle cx="{r}" cy="{r}" r="{r - 1.5}" fill="{INK}" stroke="{PAPER}" stroke-width="2"/>'
            f'{letter_path}</svg>')


def north(size: int = 26) -> str:
    """The minimap's north badge: an N drawn as a path, yellow on black."""
    k = size / 26
    n = (f'<path d="M{8.5 * k} {18.5 * k} V{7.5 * k} H{11 * k} L{15 * k} {14 * k} V{7.5 * k} H{17.5 * k} '
         f'V{18.5 * k} H{15 * k} L{11 * k} {12 * k} V{18.5 * k} Z" fill="{YELLOW}"/>')
    return badge(n, size)


def arrow(size: int = 22) -> str:
    """The car on the minimap: a red arrow pointing up with an ink edge."""
    s = size
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{s}" height="{s}">'
            f'<polygon points="{s / 2},1 {s - 2},{s - 2} {s / 2},{s * 0.72} 2,{s - 2}" fill="{RED}" '
            f'stroke="{INK}" stroke-width="1.5" stroke-linejoin="round"/></svg>')


def revs(lit: int, total: int, w: int = 150, h: int = 10) -> str:
    """The rev counter: `total` segments in a row, the first `lit` on; cream, then yellow, red at the top."""
    gap = 3
    seg = (w - gap * (total - 1)) / total
    out = []
    for k in range(total):
        if k >= lit:
            c = GREY
        elif k >= total - 2:
            c = RED
        elif k >= total - 4:
            c = YELLOW
        else:
            c = PAPER
        op = "1" if k < lit else "0.45"
        out.append(f'<rect x="{k * (seg + gap):.1f}" y="0" width="{seg:.1f}" height="{h}" fill="{c}" opacity="{op}"/>')
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">{"".join(out)}</svg>'


def bar(w: int, h: int, colour: str) -> str:
    """A plain rectangle of one colour."""
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}"><rect width="{w}" height="{h}" fill="{colour}"/></svg>'


def alpha(colour: str, a: float) -> str:
    """'#rrggbb' with an alpha from 0 to 1, as '#rrggbbaa'."""
    return f"{colour[:7]}{max(0, min(255, round(a * 255))):02x}"


def fit(panel):
    """Give a panel and every panel inside it the size its content needs, children first. manifoldx's
    layout sizes a nested panel from explicit sizes only, and a top-left root fills the window; sizes
    already set and flex spacers are kept. Returns the panel."""
    from manifoldx.gui import Panel
    from manifoldx.gui.layout import content_size

    for child in panel.children:
        if isinstance(child, Panel) and child.style_overrides.get("flex") is None:
            fit(child)
    w, h = content_size(panel.build_layout_spec())
    so = panel.style_overrides
    panel.style_overrides = {**so, "width": so.get("width") or w, "height": so.get("height") or h}
    return panel


def refit(panel):
    """Size a panel again, with the panels inside it, after its text changed."""
    from manifoldx.gui import Panel

    def drop(p):
        p.style_overrides = {k: v for k, v in p.style_overrides.items() if k not in ("width", "height")}
        for c in p.children:
            if isinstance(c, Panel) and c.style_overrides.get("flex") is None:
                drop(c)

    drop(panel)
    return fit(panel)


def photo(path, width: int, height: int) -> str:
    """A saved PNG as SVG markup, so a GUI Image can show it at width x height."""
    data = base64.b64encode(Path(path).read_bytes()).decode()
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
            f'<image width="{width}" height="{height}" href="data:image/png;base64,{data}"/></svg>')


def no_photo(width: int, height: int) -> str:
    """The grey frame a card shows for a place not photographed yet, with a camera in the middle."""
    cx, cy = width / 2, height / 2
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
            f'<rect width="{width}" height="{height}" fill="#3a3a3a"/>'
            f'<rect x="{cx - 34}" y="{cy - 20}" width="68" height="46" rx="6" fill="{GREY}"/>'
            f'<rect x="{cx - 14}" y="{cy - 30}" width="28" height="12" rx="3" fill="{GREY}"/>'
            f'<circle cx="{cx}" cy="{cy + 3}" r="14" fill="#3a3a3a"/></svg>')
