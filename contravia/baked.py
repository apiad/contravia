"""Static results kept on disk next to the data they come from: computed once, then read by every process.

The landmarks and a province's quadtree weights depend only on the data and on the code that computes them, yet
take seconds, and every builder process needs them when it starts. baked() keeps a result in a pickle beside the
data, stamped with the data file's size and modification time and with a hash of the code, and computes it again
only when either changes. The game computes them in its own process before it starts the builders, so the
builders only read them.
"""

import hashlib
import os
import pickle
from pathlib import Path


def stamp(data: Path, *code: Path) -> str:
    """What a result is computed from: the data file and the code files given, or every .py in a code folder
    given."""
    st = os.stat(data)
    h = hashlib.sha256(f"{st.st_size}:{st.st_mtime_ns}".encode())
    for path in code:
        for f in sorted(Path(path).glob("*.py")) if Path(path).is_dir() else [Path(path)]:
            h.update(f.read_bytes())
    return h.hexdigest()[:16]


def cache_dir() -> Path:
    """Where results that depend only on the code are kept: $XDG_CACHE_HOME/contravia, else ~/.cache/contravia."""
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "contravia"


def baked(path: Path, key: str, build):
    """build()'s result, kept in the pickle at path while it was made under key; computed and kept otherwise."""
    path = Path(path)
    try:
        with open(path, "rb") as f:
            held, value = pickle.load(f)
        if held == key:
            return value
    except (OSError, pickle.UnpicklingError, EOFError, ValueError):
        pass
    value = build()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as f:
        pickle.dump((key, value), f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)
    return value
