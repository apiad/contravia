"""Build the game's SVG assets: icons, minimap badges and the logo, in assets/.

Every file is plain SVG with its colours written out (no CSS, no <use>, no <text>), so manifoldx's GUI
`Image(svg=path)` draws it with thorvg at any size. The logo's letters are turned into paths from the brand
font, because an SVG renderer has no fonts loaded.

Run it again after changing an icon, a colour or the logo:
    uv run --with fonttools python assets/build.py
"""
import json
from pathlib import Path

from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.ttLib import TTFont

HERE = Path(__file__).parent
PALETTE = {"yellow": "#ffd400", "red": "#d62828", "ink": "#0b0b0b", "paper": "#f4efe4", "grey": "#9a968c"}
COLOURS = {"INK": PALETTE["ink"], "PAPER": PALETTE["paper"], "RED": PALETTE["red"], "YELLOW": PALETTE["yellow"]}

# Each icon on a 24x24 grid: (extra attributes of its group, its shapes). INK, PAPER, RED and YELLOW stand for
# the palette. The group sets the icon's fill and an ink stroke of 2.
ICONS = {
    "almendron": ('stroke-linejoin="round"', '<path d="M1.5 15.5 V12.5 L5 11.5 L8 7.5 H15 L18.5 11 L22.5 11.8 V15.5 Z"/><path d="M9 8.3 L7.3 11.3 H17 L14.6 8.3 Z" fill="PAPER"/><path d="M19 11.2 L23 9.5" fill="none"/><circle cx="6.5" cy="16" r="2.3" fill="INK"/><circle cx="17.5" cy="16" r="2.3" fill="INK"/>'),
    "lada": ('stroke-linejoin="round"', '<path d="M2 16 V11.5 L6 11 L8 7 H16 L18 11 L22 11.5 V16 Z"/><path d="M8.7 8 L7.2 11 H16.8 L15.3 8 Z" fill="PAPER"/><path d="M12 8 V11"/><circle cx="6.8" cy="16.3" r="2.2" fill="INK"/><circle cx="17.2" cy="16.3" r="2.2" fill="INK"/>'),
    "triciclo": ('stroke-linejoin="round"', '<path d="M5 5 H21 V7 H5 Z"/><path d="M6 7 V10 M20 7 V15" fill="none"/><path d="M2.5 16 L4 10.5 H9 L9.5 15 H21 V16 Z"/><circle cx="5" cy="17.5" r="2.1" fill="INK"/><circle cx="18" cy="17.5" r="2.1" fill="INK"/>'),
    "metroguagua": ('stroke-linejoin="round"', '<rect x="1" y="7.5" width="22" height="9" rx="1"/><path d="M12 7.5 V16.5" stroke-dasharray="1.5 1"/><path d="M3 10 H10.5 M13.5 10 H21" stroke="INK" stroke-width="2.4"/><circle cx="5" cy="17" r="1.9" fill="INK"/><circle cx="15" cy="17" r="1.9" fill="INK"/><circle cx="19.5" cy="17" r="1.9" fill="INK"/>'),
    "moto": ('stroke-linejoin="round"', '<circle cx="5" cy="16" r="3.3" fill="none"/><circle cx="19" cy="16" r="3.3" fill="none"/><path d="M5 16 L9.5 10.5 H15 L19 16" fill="none"/><path d="M9 10.5 L11 13.5 H15.5 L15 10.5 Z"/><path d="M15 10.5 L14 7 H17"  fill="none"/>'),
    "pasajero": ('stroke-linejoin="round"', '<circle cx="12" cy="6" r="3.4"/><path d="M5.5 21.5 V16 A6.5 6.5 0 0 1 18.5 16 V21.5 Z"/>'),
    "parada": ('stroke-linejoin="round"', '<rect x="6" y="2.5" width="12" height="8" rx="1"/><path d="M12 10.5 V21.5 M8 21.5 H16" fill="none"/><path d="M9 6.5 H15" stroke="INK"/>'),
    "destino": ('stroke-linejoin="round"', '<path d="M6 2.5 V21.5" fill="none"/><path d="M6 3.5 H19 L16 8 L19 12.5 H6 Z"/><path d="M3 21.5 H9" fill="none"/>'),
    "gasolina": ('stroke-linejoin="round"', '<rect x="4" y="3.5" width="10" height="18" rx="1"/><rect x="6.5" y="6" width="5" height="4" fill="PAPER"/><path d="M14 9 H17 L19 11 V18 A1.5 1.5 0 0 1 16 18 V14 H14" fill="none"/><path d="M2.5 21.5 H15.5" fill="none"/>'),
    "foto": ('stroke-linejoin="round"', '<path d="M2.5 7.5 H7 L8.5 5 H15.5 L17 7.5 H21.5 V19 H2.5 Z"/><circle cx="12" cy="13" r="3.8" fill="PAPER"/><circle cx="12" cy="13" r="1.5" fill="INK"/>'),
    "mapa": ('stroke-linejoin="round"', '<path d="M2.5 5.5 L8.5 3.5 L15.5 6 L21.5 4 V18.5 L15.5 20.5 L8.5 18 L2.5 20 Z"/><path d="M8.5 3.5 V18 M15.5 6 V20.5" fill="none"/>'),
    "pin": ('stroke-linejoin="round"', '<path d="M12 22 C6.5 15 5 12 5 9.2 A7 7 0 0 1 19 9.2 C19 12 17.5 15 12 22 Z"/><circle cx="12" cy="9.2" r="2.6" fill="PAPER"/>'),
    "sol": ('', '<circle cx="12" cy="12" r="4.4"/><path d="M12 1.5 V4.5 M12 19.5 V22.5 M1.5 12 H4.5 M19.5 12 H22.5 M4.6 4.6 L6.7 6.7 M17.3 17.3 L19.4 19.4 M4.6 19.4 L6.7 17.3 M17.3 6.7 L19.4 4.6" fill="none"/>'),
    "nubes": ('stroke-linejoin="round"', '<path d="M6.5 18.5 A4 4 0 0 1 6 10.6 A5.5 5.5 0 0 1 16.6 9 A4.8 4.8 0 1 1 17.5 18.5 Z"/>'),
    "lluvia": ('stroke-linejoin="round"', '<path d="M6.5 14.5 A3.8 3.8 0 0 1 6 7 A5.3 5.3 0 0 1 16.3 5.5 A4.5 4.5 0 1 1 17.5 14.5 Z"/><path d="M8 17 L7 20.5 M12.5 17 L11.5 20.5 M17 17 L16 20.5" fill="none"/>'),
    "tormenta": ('stroke-linejoin="round"', '<path d="M6.5 13.5 A3.8 3.8 0 0 1 6 6 A5.3 5.3 0 0 1 16.3 4.5 A4.5 4.5 0 1 1 17.5 13.5 Z"/><path d="M12.5 12 L9.5 17.5 H13 L11 22.5 L16 15.5 H12.5 L14 12 Z" fill="RED"/>'),
    "luna": ('stroke-linejoin="round"', '<path d="M15 2.8 A9.5 9.5 0 1 0 21.2 16 A7.5 7.5 0 0 1 15 2.8 Z"/><circle cx="18.5" cy="5" r=".9" fill="INK" stroke="none"/>'),
    "apagon": ('stroke-linejoin="round"', '<path d="M8 15.5 C5 13.5 5 8.5 7.5 6 A6.3 6.3 0 0 1 18.2 10.5 C18.2 12.8 17 14.3 16 15.5 Z"/><path d="M8.5 18 H15.5 M9.5 21 H14.5" fill="none"/><path d="M3 3 L21 21" stroke="RED" stroke-width="2.6" fill="none"/>'),
    "trafico": ('stroke-linejoin="round"', '<rect x="7.5" y="1.5" width="9" height="21" rx="1.5"/><circle cx="12" cy="6" r="2" fill="RED"/><circle cx="12" cy="12" r="2" fill="PAPER"/><circle cx="12" cy="18" r="2" fill="PAPER"/>'),
    "reloj": ('', '<circle cx="12" cy="12" r="9.5"/><path d="M12 6.5 V12 L15.5 14.5" fill="none" stroke-linecap="round"/>'),
    "velocidad": ('stroke-linejoin="round"', '<path d="M2.5 17 A9.5 9.5 0 0 1 21.5 17 Z"/><path d="M12 17 L17 9.5" fill="none" stroke="RED" stroke-width="2.4" stroke-linecap="round"/><circle cx="12" cy="17" r="1.6" fill="INK"/>'),
    "brujula": ('', '<circle cx="12" cy="12" r="9.5"/><path d="M12 4.5 L14.6 12 H9.4 Z" fill="RED"/><path d="M12 19.5 L14.6 12 H9.4 Z" fill="PAPER"/>'),
    "ajustes": ('stroke-linejoin="round"', '<path d="M10.2 2 H13.8 L14.4 4.8 L16.6 6 L19.3 5 L21.1 8.1 L19 10 V12.4 L21.1 14.3 L19.3 17.4 L16.6 16.4 L14.4 17.6 L13.8 20.4 H10.2 L9.6 17.6 L7.4 16.4 L4.7 17.4 L2.9 14.3 L5 12.4 V10 L2.9 8.1 L4.7 5 L7.4 6 L9.6 4.8 Z"/><circle cx="12" cy="11.2" r="3" fill="PAPER"/>'),
    "sonido": ('stroke-linejoin="round"', '<path d="M2.5 9 H7 L12.5 4.5 V19.5 L7 15 H2.5 Z"/><path d="M15.5 9 A4 4 0 0 1 15.5 15 M18 6 A8 8 0 0 1 18 18" fill="none" stroke-linecap="round"/>'),
    "mando": ('stroke-linejoin="round"', '<path d="M4 8 H20 A2.5 2.5 0 0 1 22.4 11 L21 18 A2 2 0 0 1 17.6 19 L15.5 16 H8.5 L6.4 19 A2 2 0 0 1 3 18 L1.6 11 A2.5 2.5 0 0 1 4 8 Z"/><path d="M5.5 12 H9.5 M7.5 10 V14" fill="none" stroke="INK"/><circle cx="16" cy="11" r="1" fill="INK" stroke="none"/><circle cx="18" cy="13" r="1" fill="INK" stroke="none"/>'),
    "contravia": ('', '<circle cx="12" cy="12" r="10" fill="RED"/><rect x="6" y="10" width="12" height="4" fill="PAPER" stroke="none"/>'),
    "peso": ('stroke-linejoin="round"', '<rect x="1.5" y="5.5" width="21" height="13" rx="1"/><circle cx="12" cy="12" r="3.4" fill="PAPER"/><path d="M4.5 9 V15 M19.5 9 V15" fill="none"/>'),
    "casa": ('stroke-linejoin="round"', '<path d="M2.5 11 L12 3 L21.5 11 V21.5 H2.5 Z"/><rect x="9" y="14" width="6" height="7.5" fill="PAPER"/>'),
}
# Icons filled with paper (neutral); every other icon is filled with yellow (something you can use).
NEUTRAL = {"pasajero", "mapa", "nubes", "lluvia", "luna", "reloj", "brujula", "casa", "pin", "parada", "lada", "moto"}


def colour(text: str) -> str:
    for name, hex_ in COLOURS.items():
        text = text.replace(f'"{name}"', f'"{hex_}"')
    return text


def icon(name: str, fill: str, stroke: str, width: float) -> str:
    """One icon as a group: the given fill and stroke, round caps and joins."""
    attrs, body = ICONS[name]
    return (f'<g fill="{fill}" stroke="{stroke}" stroke-width="{width}" stroke-linecap="round" '
            f'stroke-linejoin="round" {colour(attrs)}>{colour(body)}</g>')


def svg(size: float, body: str, view: str | None = None) -> str:
    view = view or f"0 0 {size} {size}"
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="{view}">'
            f"{body}</svg>\n")


def build_icons() -> int:
    """icons/<name>.svg on light ground, and icons/badge/<name>.svg for the minimap: a paper disc with an ink
    ring and the icon inside, so it reads on the map's greys and greens."""
    (HERE / "icons" / "badge").mkdir(parents=True, exist_ok=True)
    for name in ICONS:
        fill = PALETTE["paper"] if name in NEUTRAL else PALETTE["yellow"]
        (HERE / "icons" / f"{name}.svg").write_text(svg(24, icon(name, fill, PALETTE["ink"], 2)))
        badge = (f'<circle cx="16" cy="16" r="14.5" fill="{PALETTE["paper"]}" stroke="{PALETTE["ink"]}" stroke-width="2.5"/>'
                 f'<g transform="translate(6.4 6.4) scale(0.8)">{icon(name, fill, PALETTE["ink"], 2)}</g>')
        (HERE / "icons" / "badge" / f"{name}.svg").write_text(svg(32, badge))
    return len(ICONS)


def word_paths(text: str, font_path: Path, size: float) -> tuple[str, float, float]:
    """The text as SVG paths in a y-down box `size` high, with its advance width and the glyph-to-box scale."""
    font = TTFont(font_path)
    glyphs, cmap, upm = font.getGlyphSet(), font.getBestCmap(), font["head"].unitsPerEm
    ascent, descent = font["hhea"].ascent, font["hhea"].descent
    scale, x, parts = size / (ascent - descent), 0.0, []
    for ch in text:
        name = cmap[ord(ch)]
        pen = SVGPathPen(glyphs)
        glyphs[name].draw(pen)
        parts.append(f'<path transform="translate({x:.2f} {ascent * scale:.2f}) scale({scale:.5f} {-scale:.5f})" '
                     f'd="{pen.getCommands()}"/>')
        x += glyphs[name].width * scale
    return "".join(parts), x, scale


def build_logo() -> None:
    """logo/sign.svg (the no-entry sign), logo/word.svg (CONTRAVÍA) and logo/contravia.svg (both, side by side),
    drawn as on the poster: yellow letters skewed 6 degrees, an ink outline and a red drop shadow."""
    (HERE / "logo").mkdir(exist_ok=True)
    p, k, r, y = PALETTE["paper"], PALETTE["ink"], PALETTE["red"], PALETTE["yellow"]
    sign = (f'<circle cx="50" cy="50" r="48" fill="{k}"/><circle cx="50" cy="50" r="43" fill="{p}"/>'
            f'<circle cx="50" cy="50" r="37" fill="{r}"/><rect x="24" y="42" width="52" height="16" fill="{p}"/>')
    (HERE / "logo" / "sign.svg").write_text(svg(100, sign))
    letters, w, gscale = word_paths("CONTRAVÍA", HERE / "fonts" / "NotoSans-CondensedBlackItalic.ttf", 200)
    out, sh, top = 8, 9, 30  # outline width, shadow offset and headroom for the accent, in units of the 200-high word
    def layer(fill, stroke=None, dx=0):
        # the letters are scaled paths, so the stroke width is given in the font's units
        s = f' stroke="{stroke}" stroke-width="{2 * out / gscale:.0f}" stroke-linejoin="round"' if stroke else ""
        return f'<g transform="translate({dx + 24} {dx + top}) skewX(-6)" fill="{fill}"{s}>{letters}</g>'
    word = layer(r, r, sh + out) + layer(k, k, out) + layer(y, None, out)
    W, H = w + 2 * out + sh + 60, 200 + 2 * out + sh + top
    view = f"0 0 {W:.0f} {H:.0f}"
    (HERE / "logo" / "word.svg").write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W:.0f}" height="{H:.0f}" viewBox="{view}">{word}</svg>\n')
    full = (f'<g transform="translate(0 {(H - 150) / 2:.1f}) scale(1.5)">{sign}</g>'
            f'<g transform="translate(180 0)">{word}</g>')
    FW = W + 180
    (HERE / "logo" / "contravia.svg").write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{FW:.0f}" height="{H:.0f}" viewBox="0 0 {FW:.0f} {H:.0f}">{full}</svg>\n')


def main() -> None:
    n = build_icons()
    build_logo()
    (HERE / "palette.json").write_text(json.dumps(PALETTE, indent=1) + "\n")
    print(f"{n} icons, {n} badges, 3 logo files, palette.json")


if __name__ == "__main__":
    main()
