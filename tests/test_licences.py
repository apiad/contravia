"""The repository declares a licence for every path (REUSE.toml) and ships each licence's text."""

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_every_licence_named_in_reuse_toml_has_its_text():
    ids = set(re.findall(r'SPDX-License-Identifier = "([^"]+)"', (ROOT / "REUSE.toml").read_text()))
    assert ids == {"GPL-3.0-or-later", "CC-BY-NC-SA-4.0", "CC-BY-4.0", "OFL-1.1", "ODbL-1.0",
                   "LicenseRef-Contravia-Mods"}
    for i in ids:
        assert (ROOT / "LICENSES" / f"{i}.txt").stat().st_size > 0, i


def test_the_root_licence_is_the_gpl_and_pyproject_says_so():
    assert (ROOT / "LICENSE").read_bytes() == (ROOT / "LICENSES" / "GPL-3.0-or-later.txt").read_bytes()
    assert tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["license"] == "GPL-3.0-or-later"


def test_the_package_copies_the_licence_files():
    build = (ROOT / "packaging" / "build.sh").read_text()
    copy = next(line for line in build.splitlines() if line.lstrip().startswith("rsync") and "/contravia" in line)
    for name in ("LICENSE", "LICENSES", "MODS.md", "TRADEMARKS.md"):
        assert f'"$REPO/{name}"' in copy, name
