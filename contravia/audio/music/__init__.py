"""Music by biome: five styles, composed and rendered ahead by the music process (player.py)."""

from contravia.audio.music.base import Style, seed_of
from contravia.audio.music.bolero import Bolero
from contravia.audio.music.changui import Changui
from contravia.audio.music.guaguanco import Guaguanco
from contravia.audio.music.guajira import Guajira
from contravia.audio.music.son import Son

STYLES = {"son": Son, "guaguanco": Guaguanco, "guajira": Guajira, "changui": Changui, "bolero": Bolero}

__all__ = ["STYLES", "Style", "seed_of"]
