"""The menus' and the camera's sounds."""

from types import SimpleNamespace

import numpy as np
from euterpe.sound import render

from contravia.audio.ui import KINDS, ui


def test_each_ui_sound_is_short_and_audible():
    for k in KINDS:
        u = ui(k, seed=1)
        x = render(u, u.duration)
        assert u.duration < 1.5 and np.sqrt(np.mean(x**2)) > 1e-3, k


def test_the_menu_asks_for_a_sound_on_each_key():
    from contravia.menu import Menu

    asked = []
    m = Menu.__new__(Menu)
    m.current = "main"
    m.screens = {"main": SimpleNamespace(key=lambda k: None)}
    m.game = SimpleNamespace(sound=SimpleNamespace(ui=asked.append), mode="menu", curtain=None)
    for key in ("ArrowDown", "Enter", "Escape"):
        m.press(key)
    assert asked == ["move", "choose", "back"]
