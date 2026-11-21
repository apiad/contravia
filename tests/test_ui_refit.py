"""ui.refit inside a pipelined frame: the text and children set before it land at apply time, and so does the
measure, so the panel is sized for its new content (the map's panel was drawn narrow, its rows past the edge)."""

import manifoldx as mx
from manifoldx.gui import Panel, Text

from contravia import ui


def _width(panel) -> float:
    return panel.style_overrides["width"]


def test_refit_in_a_pipelined_frame_measures_the_new_content():
    engine = mx.Engine("refit")
    title = Text("Granma", style_overrides={"font": ui.BOLD, "font_size": 24})
    panel = ui.fit(Panel([title]))
    narrow = _width(panel)
    rows = [Text("Basílica Santuario de la Caridad del Cobre", style_overrides={"font": ui.BOLD, "font_size": 14})]
    with engine._defer_calls():
        panel.children = [title, *rows]
        ui.refit(panel)
    engine.commands.execute(engine.store)
    assert len(panel.children) == 2
    assert _width(panel) > narrow * 1.5


def test_refit_outside_a_pipeline_acts_at_once():
    title = Text("Granma", style_overrides={"font": ui.BOLD, "font_size": 24})
    panel = ui.fit(Panel([title]))
    narrow = _width(panel)
    panel.children = [title, Text("Basílica Santuario de la Caridad del Cobre",
                                  style_overrides={"font": ui.BOLD, "font_size": 14})]
    ui.refit(panel)
    assert _width(panel) > narrow * 1.5
