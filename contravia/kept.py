"""Results that survive a save under contravia dev: what takes seconds to build and depends only on its inputs.

manifoldx's hot reload keeps engine.state from one version of the code to the next, and nothing else. kept()
holds a result there under a name, with a stamp of what it was built from, and builds it again when the stamp
changes. The stamp includes source(), the modification time of the file whose code builds the result, so a save
to that file builds it afresh. A kept result is plain data or an object of an installed library, never an
instance of contravia's own classes, whose code a reload replaces. Under contravia play each engine has its own state,
so kept() builds each result once, as before.
"""

import os


class _Shelf:
    """The kept results, under one key of engine.state; its short repr keeps them out of `manifoldx state`."""

    def __init__(self):
        self.items = {}

    def __repr__(self):
        return f"<kept: {', '.join(sorted(self.items))}>"


def source(path: str) -> int:
    """A source file's modification time, as part of a stamp."""
    return os.stat(path).st_mtime_ns


def kept(state, name: str, stamp, build):
    """build()'s result, kept in state under name until stamp changes."""
    shelf = state.setdefault("kept", _Shelf()).items
    held = shelf.get(name)
    if held is None or held[0] != stamp:
        held = shelf[name] = (stamp, build())
    return held[1]
