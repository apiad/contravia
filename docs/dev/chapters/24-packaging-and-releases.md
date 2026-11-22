# Packaging and releases {#packaging-and-releases}

A player should not install Python 3.14, its libraries and a graphics engine, and should not be asked to build an island from 16 province packs. The release is one file for Linux and one for Windows, each carrying its own interpreter and the island's sources; the first run builds the island in the player's data folder. This chapter follows that file from `packaging/build.sh` to the release page, and then the licences every path must declare.

## What a release carries

| Carried | Where it is | Why |
| --- | --- | --- |
| The launcher | The first part of the executable | Starts the game and unpacks the rest |
| Python and the libraries | A zip appended to the launcher, under `python/` and `site/` | A free-threaded Python 3.14 and the wheels for the platform |
| The game's code and assets | The same zip, under `app/` | `contravia/`, `assets/`, the poster panels, the licences and the mod terms |
| The island's sources | `app/data/cuba-src/` | The 16 province `.src` files and the six island-wide files |

The built island is not carried. `data/` is never committed and never packaged: the release ships the packs a province is built from, and the first run builds `cuba/` in the player's data folder. That first run takes about 19 minutes and 6 GB on the reference machine with three provinces at once. A player whose `cuba/build.json` names another built-city `FORMAT` (7 today) has no island the game will use, so the first run builds it again.

## One file: the launcher and its zip

`packaging/launcher/main.go` is a small Go program. The game's file has three parts, in order: the launcher, the zip, and the zip's length as 8 little-endian bytes at the very end. The launcher finds the length, opens the zip as a section of its own file and reads the rest.

On the first run of a version, the launcher unpacks `python/`, `site/` and `app/` into `<data home>/runtime/<version>` through a folder named `<version>.part`, then renames it whole. It removes every other runtime in that folder, so an update does not leave the old one on disk. After that it runs

```text
python -m contravia.cli [your arguments]
```

from the `app/` folder, with `PYTHONPATH` set to `app` and `site` and `PYTHONNOUSERSITE=1` so a user's site-packages cannot change the game. With no arguments it runs `contravia play`. Everything else is passed through: `contravia --help`, `contravia build`, `contravia bench`.

The data home is the same for the runtime and for the built island: `CONTRAVIA_HOME` when it is set, else `%LOCALAPPDATA%\Contravia` on Windows, else `$XDG_DATA_HOME/contravia`, else `~/.local/share/contravia`. `contravia/firstrun.py` reads the same names, so `runtime/` and `cuba/` sit side by side.

<figure>
<svg viewBox="0 0 640 430" width="100%" role="img" aria-labelledby="package-title" font-family="ui-sans-serif, system-ui, sans-serif">
<title id="package-title">The single-file game: the launcher, the zip and its length</title>
<defs>
  <marker id="pkg-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#0b0b0b"/>
  </marker>
</defs>

<text x="14" y="24" font-size="11" font-weight="700" fill="#d62828" letter-spacing="1.4">ONE FILE PER OPERATING SYSTEM</text>

<rect x="14" y="40" width="160" height="190" rx="3" fill="#fff8d6" stroke="#0b0b0b" stroke-width="1.1"/>
<text x="28" y="62" font-size="12.5" font-weight="700" fill="#0b0b0b">launcher</text>
<text x="28" y="79" font-size="10.5" fill="#555">Go, static</text>
<text x="28" y="95" font-size="10.5" fill="#555">finds the length</text>
<text x="28" y="111" font-size="10.5" fill="#555">opens the zip</text>
<text x="28" y="127" font-size="10.5" fill="#555">unpacks a version</text>
<text x="28" y="143" font-size="10.5" fill="#555">runs contravia.cli</text>
<text x="28" y="175" font-family="ui-monospace, monospace" font-size="10" fill="#d62828">app/ · site/ · python/</text>
<text x="28" y="193" font-size="10" fill="#777">into runtime/&lt;version&gt;</text>
<text x="28" y="209" font-size="10" fill="#777">then deletes the others</text>

<rect x="184" y="40" width="300" height="190" rx="3" fill="#f6f4ef" stroke="#0b0b0b" stroke-width="1.1"/>
<text x="198" y="62" font-size="12.5" font-weight="700" fill="#0b0b0b">the zip</text>
<text x="198" y="84" font-family="ui-monospace, monospace" font-size="10.5" fill="#555">python/</text>
<text x="272" y="84" font-size="10" fill="#777">3.14.7, free-threaded, stripped</text>
<text x="198" y="104" font-family="ui-monospace, monospace" font-size="10.5" fill="#555">site/</text>
<text x="272" y="104" font-size="10" fill="#777">the wheels for cp314t and the platform</text>
<text x="198" y="124" font-family="ui-monospace, monospace" font-size="10.5" fill="#555">app/contravia/</text>
<text x="272" y="124" font-size="10" fill="#777">the code</text>
<text x="198" y="144" font-family="ui-monospace, monospace" font-size="10.5" fill="#555">app/assets/, app/branding/poster/</text>
<text x="198" y="164" font-family="ui-monospace, monospace" font-size="10.5" fill="#555">app/data/cuba-src/</text>
<text x="272" y="164" font-size="10" fill="#777">16 + 6 .src files</text>
<text x="198" y="184" font-family="ui-monospace, monospace" font-size="10.5" fill="#555">app/LICENSE, LICENSES/, MODS.md</text>
<text x="198" y="206" font-size="10" fill="#777">no symlinks, no __pycache__, no .dist-info/RECORD</text>

<rect x="494" y="40" width="132" height="190" rx="3" fill="#f6f4ef" stroke="#0b0b0b" stroke-width="1.1"/>
<text x="508" y="62" font-size="12.5" font-weight="700" fill="#0b0b0b">8 bytes</text>
<text x="508" y="80" font-size="10.5" fill="#555">the zip's length,</text>
<text x="508" y="96" font-size="10.5" fill="#555">little-endian</text>
<text x="508" y="120" font-size="10" fill="#777">at the end of the</text>
<text x="508" y="136" font-size="10" fill="#777">launcher + zip</text>
<text x="508" y="166" font-family="ui-monospace, monospace" font-size="10" fill="#d62828">contravia</text>
<text x="508" y="182" font-family="ui-monospace, monospace" font-size="10" fill="#d62828">contravia.exe</text>

<path d="M174 135 H180" stroke="#0b0b0b" stroke-width="1.2" marker-end="url(#pkg-arrow)" fill="none"/>
<path d="M484 135 H490" stroke="#0b0b0b" stroke-width="1.2" marker-end="url(#pkg-arrow)" fill="none"/>

<rect x="14" y="266" width="612" height="94" rx="3" fill="#f6f4ef" stroke="#0b0b0b" stroke-width="1.1"/>
<text x="28" y="287" font-size="12.5" font-weight="700" fill="#0b0b0b">on the player's machine</text>
<text x="28" y="306" font-family="ui-monospace, monospace" font-size="10.5" fill="#555">&lt;data home&gt;/runtime/&lt;version&gt;/app · site · python</text>
<text x="28" y="324" font-family="ui-monospace, monospace" font-size="10.5" fill="#555">&lt;data home&gt;/cuba/ · build.json</text>
<text x="28" y="344" font-size="10" fill="#777">the first run builds the island from app/data/cuba-src beside the runtime, and writes build.json when it is whole</text>

<text x="14" y="392" font-size="10.5" fill="#555">The launcher is the only compiled part. Everything after it is data: a zip the game's own interpreter reads at the</text>
<text x="14" y="407" font-size="10.5" fill="#555">next run. Two targets, x86_64-unknown-linux-gnu and x86_64-pc-windows-msvc, from the same build script.</text>
</svg>
<figcaption>One executable per platform: the launcher, then the zip with the interpreter, the wheels and the game, then the zip's length.</figcaption>
</figure>

## What build.sh does

`packaging/build.sh OUT [WORK]` builds both files. `OUT` holds the results; `WORK` (default `/tmp/contravia-pack`) holds the downloads and the trees. The script needs `go`, `uv`, `curl`, `zip` and `git`, and `PIP_PYTHON`, a free-threaded Python 3.14 with `pip`, because `uv` cannot pick `cp314t` wheels for another platform and `pip` can with `--abi`.

The steps are:

1. Check the sources. `CUBA_SRC` (default `data/cuba-src`) must hold all 16 province sources at the `contravia.source.VERSION` the game reads, 5 today, and the side files it names. A missing or stale one stops the build with the command that writes it.
2. Export the locked runtime requirements with `uv export --frozen --no-dev`, without the project itself, and build the manifoldx wheel once.
3. For each target, download python-build-standalone's free-threaded Python 3.14.7 (`PBS` 20260807), strip what the game never loads (tests, Tk, pip, static libraries, `__pycache__`), and install the wheels into `site/` with `pip` for the target's platform and the `cp314t` ABI.
4. Copy `contravia/`, `assets/`, `LICENSE`, `LICENSES/`, `MODS.md`, `TRADEMARKS.md` and `branding/poster/panels` into `app/`, and the sources into `app/data/cuba-src/`. A symlink anywhere stops the build, since the launcher writes files and not links.
5. Zip `python`, `site` and `app` at level 9, build the Go launcher for the target with `-trimpath -ldflags "-s -w -X main.version=$VERSION"`, and append the zip to the launcher.
6. Append the zip's length as 8 little-endian bytes, make the file executable, and print its size in MB.

`VERSION` is the release's tag when the workflow sets it, and the commit's short SHA otherwise. The launcher uses it as the runtime's folder name, so a new build never runs on top of an old runtime.

The sources are checked with the same loaders the game uses: `source.load`, `coast.load`, `shore.load`, `regions.Regions.load`, `causeway.load` and `cover.load`. The check loads five of the six island-wide files; `water.src` is not in it, and the release workflow's smoke test is what counts all six in the package. The build refuses to package a province or one of the side files it loads at another version, which is what keeps a release from shipping an island whose streets the game cannot read.

## The sources release

The province packs are not in git and are not built by the release's own checkout. They are packed once from the OpenStreetMap extract and published as their own release:

```bash
contravia source cuba-latest.osm.pbf --dem tiles --worldcover tiles --out data/cuba-src
contravia cover --out data/cuba-src
```

`contravia source` writes one `<slug>.src` per province, a zstd file at level 19 holding the province's `Raw` and its cleaned relief as `int16` decimetres. When it packs the whole island, it also writes `coast.src`, `shore.src`, `regions.src`, `causeways.src` and `water.src` beside them; `contravia cover` writes `cover.src`, because it downloads WorldCover tiles over the coast and deletes each once read. The pack format is `contravia.source.VERSION`: 5 today, after 2 added road names, places and consejos populares, 3 stopped land from becoming a park, 4 added the satellite's built-up grid, and 5 put every province's relief on one lattice.

The packs are published as `sources-v<VERSION>`, a `cuba-src.tar.zst` of the folder:

```bash
gh release create sources-v5 cuba-src.tar.zst --title "Contravía sources v5" --latest=false
```

`--latest=false` is required. The landing page's two download buttons point at `releases/latest/download/contravia` and `releases/latest/download/contravia.exe`, so the latest release must stay a game release. Pack the sources again whenever `source.VERSION` changes, because a game of one version refuses a source of another.

The package's own counts are the ones the build and the smoke test check:

```python {run}
from contravia import causeway, coast, cover, regions, shore, source, water
from contravia.geo import PROVINCES

sides = (coast, shore, regions, causeway, water, cover)
print(f"{len(PROVINCES)} province sources at .src version {source.VERSION}")
print(f"{len(sides)} island-wide files: {', '.join(module.NAME for module in sides)}")
```

## The release workflow

`.github/workflows/release.yml` runs on a pushed tag `v*`, and by hand with `workflow_dispatch`. A tag publishes a release; a manual run builds the same files as workflow artifacts, kept for 7 days, to try the packaging.

The job checks out the tag, installs Go from `packaging/launcher/go.mod` and `uv`, and installs a free-threaded Python to set `PIP_PYTHON`. It reads `source.VERSION` from `contravia/source.py` and downloads `sources-v<VERSION>` into `data/`, then runs `packaging/build.sh out`. `VERSION` is the tag's name for a tag and the commit's SHA for a manual run.

Before publishing, it checks the Linux file the way a player would find it. It lists the zip and counts 16 province sources and 6 island-wide ones; it copies one province source into an empty folder, builds it with `out/contravia build --source ... --home ...`, and requires `build.json`, `<province>.npz` and `index.npz` to appear. That smoke test is the one check that the file is a game and not a pile of code: the province build runs the same first-run path a player's machine runs.

A tag then publishes with `gh release create "$GITHUB_REF_NAME" out/contravia out/contravia.exe --title "Contravía $GITHUB_REF_NAME" --generate-notes`. The landing page follows that release, so its two files must be there.

## Licences and REUSE

The repository is public and every path declares its licence in `REUSE.toml`. `uvx reuse lint` runs in CI and fails on a path with no block, which is why a new top-level folder needs one.

| Layer | Licence | Where it applies |
| --- | --- | --- |
| Code | GPL-3.0-or-later | `contravia/**`, `tests/**`, `bench/**`, `packaging/**`, `site/**`, the workflows and the build files |
| Map data | ODbL-1.0 | `data/**`, `contravia/places.json`, `contravia/towns.json`; the credit is OpenStreetMap contributors, Copernicus DEM (DLR e.V., Airbus) and ESA WorldCover |
| Art, text and docs | CC-BY-NC-SA-4.0 | `assets/**`, `branding/**`, `docs/**`, `know-how/**`, `README.md`, `AGENTS.md`, `CONTRIBUTING.md`, `TRADEMARKS.md` |
| Fonts | OFL-1.1 | The Noto faces in `assets/fonts/**` |
| Sky tables | CC-BY-NC-SA-4.0 and CC-BY-4.0 | The star and Milky Way arrays, and the climate table, which holds ERA5 and NOAA data |
| Mods | LicenseRef-Contravia-Mods | `MODS.md` and its licence text: an additional permission over the core assets that lets anyone sell mods under four conditions |
| The name and logo | Not a licence | `TRADEMARKS.md`: forks and mods ship under another name and without the logo |

`LICENSES/` holds the texts, and `pyproject.toml` names `LICENSE` and `LICENSES/*.txt` as the wheel's licence files, so a package carries them. The single-file build copies `LICENSE`, `LICENSES/`, `MODS.md` and `TRADEMARKS.md` into `app/`, beside the code they cover. The code is open source; the art and text are source-available and non-commercial; the map data is open under ODbL; the name is not granted by any of them.

The sky tables follow their sources rather than the code: the star catalogue and the Milky Way are public-domain data the project processed, and the climate chain is a modified Copernicus and NOAA product. `contravia/credits.py` is what the game shows for that credit.

## What a merge and a release need

A merge to main needs green tests and nothing else; `fastci` runs the tests the diff can break and `reuse lint` checks the licences. A release is heavier: the full `contravia bench` over the whole route, about 15 minutes, and a `contravia streets-check --fresh` over the island run before it, because a kept check from before the last change is not a check. [Checking the island and measuring it](#checking-and-measuring) holds those gates.

## The modules behind this

| Piece | Module | What it owns |
| --- | --- | --- |
| The single file | `packaging/launcher/main.go`, `packaging/build.sh` | The launcher, the zip, the length, the two targets |
| The first run | `contravia/firstrun.py`, `contravia/buildscreen.py` | The home, `build.json`, `FORMAT`, the province steps |
| The sources | `contravia/source.py` | `VERSION`, level 19, one pack per province |
| The side files | `contravia/coast.py`, `shore.py`, `regions.py`, `causeway.py`, `water.py`, `cover.py` | The six island-wide `.src` files and their versions |
| The release | `.github/workflows/release.yml`, `.github/workflows/test.yml`, `.github/workflows/pages.yml` | The tag build, the smoke test, the checks, the books and the landing |
| The licences | `REUSE.toml`, `LICENSES/`, `LICENSE`, `MODS.md`, `TRADEMARKS.md` | The per-path blocks, the texts, the mod permission, the name |
