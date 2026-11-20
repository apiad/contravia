#!/usr/bin/env bash
# Build the single-file game for Linux and Windows (x86-64): the launcher with the free-threaded Python, the
# runtime wheels, the game's code, assets, poster panels and the 16 province sources appended as a zip.
#
#   packaging/build.sh OUT [WORK]
#
# The province sources come from CUBA_SRC (default data/cuba-src): one <slug>.src per province, at the version the
# game reads. They are not in git; make them with `contravia source <cuba-latest.osm.pbf> --dem DIR`.
#
# Needs go, uv, curl, zip, git on PATH, and PIP_PYTHON: a free-threaded 3.14 with pip (uv cannot pick
# cp314t wheels for another platform; pip can, with --abi). WORK (default /tmp/contravia-pack) holds the downloads and the trees.
set -euo pipefail

OUT=$(realpath -m "${1:?usage: build.sh OUT [WORK]}")
WORK=$(realpath -m "${2:-/tmp/contravia-pack}")
REPO=$(cd "$(dirname "$0")/.." && pwd)
PY=3.14.7
PBS=20260807
VERSION=${VERSION:-$(git -C "$REPO" rev-parse --short HEAD)}
mkdir -p "$OUT" "$WORK"

# The province sources: all 16, the coast and the cover, each one the version this game reads, or no package.
SRC=$(realpath -m "${CUBA_SRC:-$REPO/data/cuba-src}")
uv run --project "$REPO" --frozen --quiet python -c '
import sys
from pathlib import Path
from contravia.geo import PROVINCES
from contravia import source
bad = []
for name, slug in PROVINCES:
    path = Path(sys.argv[1]) / f"{slug}.src"
    try:
        source.load(path)
    except (OSError, ValueError) as e:
        bad.append(f"  {name}: {e}")
from contravia import coast, cover, shore
try:
    coast.load(Path(sys.argv[1]) / coast.NAME)
except (OSError, ValueError) as e:
    bad.append(f"  the coast: {e}; write it with: contravia coast <cuba-latest.osm.pbf> --out {sys.argv[1]}")
try:
    shore.load(Path(sys.argv[1]) / shore.NAME)
except (OSError, ValueError) as e:
    bad.append(f"  the shore: {e}; write it with: contravia shore <cuba-latest.osm.pbf> --out {sys.argv[1]}")
try:
    cover.load(Path(sys.argv[1]) / cover.NAME)
except (OSError, ValueError) as e:
    bad.append(f"  the cover: {e}; write it with: contravia cover --out {sys.argv[1]}")
if bad:
    sys.exit(f"{sys.argv[1]}: {len(bad)} of {len(PROVINCES)} province sources missing or stale; "
             "pack them with: contravia source <cuba-latest.osm.pbf> --dem DIR --out " + sys.argv[1] + "\n" + "\n".join(bad))
print(f"{sys.argv[1]}: {len(PROVINCES)} province sources, version {source.VERSION}")
' "$SRC"

# The runtime dependencies, as locked, without the project itself; manifoldx comes from git, as one wheel.
PIP="${PIP_PYTHON:?set PIP_PYTHON to a free-threaded python3.14 with pip} -m pip"
uv export --project "$REPO" --frozen --no-dev --no-hashes --no-emit-project --format requirements.txt \
    | grep -v '^ *#' > "$WORK/requirements.txt"
rm -rf "$WORK/wheels" && mkdir -p "$WORK/wheels"
$PIP wheel --quiet --no-deps -w "$WORK/wheels" "$(grep '^manifold-gfx' "$WORK/requirements.txt")"

for target in linux windows; do
    case $target in
        linux) triple=x86_64-unknown-linux-gnu; goos=linux; exe=contravia; drop='colorama|rubicon'
               platforms="--platform manylinux_2_28_x86_64 --platform manylinux_2_17_x86_64 --platform manylinux2014_x86_64 --platform linux_x86_64" ;;
        windows) triple=x86_64-pc-windows-msvc; goos=windows; exe=contravia.exe; drop='rubicon'
                 platforms="--platform win_amd64" ;;
    esac
    tree="$WORK/$target"
    rm -rf "$tree"
    mkdir -p "$tree/app" "$tree/site"

    # Python: python-build-standalone's free-threaded build, stripped, without what the game never loads.
    tarball="$WORK/python-$PY-$target.tar.gz"
    [ -f "$tarball" ] || curl -fsSL -o "$tarball" \
        "https://releases.astral.sh/github/python-build-standalone/releases/download/$PBS/cpython-$PY%2B$PBS-$triple-freethreaded-install_only_stripped.tar.gz"
    tar -xzf "$tarball" -C "$tree"
    lib=$( [ $target = linux ] && echo "$tree/python/lib/python3.14t" || echo "$tree/python/Lib" )
    rm -rf "$lib"/{test,tkinter,idlelib,ensurepip,turtledemo,turtle.py,site-packages/pip*} \
        "$tree/python"/{include,share,tcl,lib/tcl9*,lib/tk9*,lib/itcl*,lib/thread*} \
        "$tree/python"/{Scripts,libs,include}
    if [ $target = linux ]; then  # bin/python3.14t is static: libpython, the other names and Tcl/Tk go
        find "$tree/python/bin" -mindepth 1 ! -name python3.14t -delete
        rm -rf "$tree/python/lib"/{libpython*,libtcl*,pkgconfig} "$lib"/config-*
    fi
    find "$tree/python" -name __pycache__ -prune -exec rm -rf {} +

    # The wheels for this platform, into site/.
    grep -vE "^(manifold-gfx|$drop)" "$WORK/requirements.txt" | sed 's/ *;.*//' > "$WORK/req-$target.txt"
    $PIP install --quiet --no-deps --target "$tree/site" $platforms --implementation cp --python-version 3.14 \
        --abi cp314t --only-binary :all: -r "$WORK/req-$target.txt" "$WORK"/wheels/manifold_gfx-*.whl
    find "$tree/site" -name __pycache__ -prune -exec rm -rf {} +
    rm -rf "$tree/site"/bin "$tree/site"/*.dist-info/RECORD

    # The game.
    rsync -a --exclude __pycache__ "$REPO/contravia" "$REPO/assets" "$tree/app/"
    mkdir -p "$tree/app/branding/poster" "$tree/app/data/cuba-src"
    rsync -a "$REPO/branding/poster/panels" "$tree/app/branding/poster/"
    cp "$SRC"/*.src "$tree/app/data/cuba-src/"

    # The launcher, then the zip, then the zip's length. The launcher writes files only, not links.
    if [ -n "$(find "$tree" -type l)" ]; then echo "links in $tree:" && find "$tree" -type l && exit 1; fi
    (cd "$tree" && rm -f "$WORK/$target.zip" && zip -q -9 -r -X "$WORK/$target.zip" python site app)
    (cd "$REPO/packaging/launcher" && CGO_ENABLED=0 GOOS=$goos GOARCH=amd64 \
        go build -trimpath -ldflags "-s -w -X main.version=$VERSION" -o "$WORK/launcher-$target" .)
    cat "$WORK/launcher-$target" "$WORK/$target.zip" > "$OUT/$exe"
    python3 -c "import struct, sys; sys.stdout.buffer.write(struct.pack('<Q', $(stat -c %s "$WORK/$target.zip")))" >> "$OUT/$exe"
    chmod +x "$OUT/$exe"
    echo "$OUT/$exe: $(( $(stat -c %s "$OUT/$exe") / 1000000 )) MB"
done
