# Contributing to Contravía

Thank you for helping. Bug reports, fixes, buildings, cars, missions and data are all welcome.

## What your contribution is licensed under

Contravía has four layers, each with its own licence (`REUSE.toml` maps every path to one):

| Layer | Licence | Where |
|-------|---------|-------|
| Code | GPL-3.0-or-later | `contravia/`, `havana/`, `tests/`, `packaging/` and the build files |
| Map data | ODbL-1.0 | `data/`, `contravia/places.json`, `contravia/towns.json` |
| Core assets and docs | CC-BY-NC-SA-4.0 | `assets/`, `branding/`, `docs/`, `know-how/` |
| Paid extras (mods and assets, never mechanics or essential content) | all rights reserved | not in this repository |

Inbound equals outbound: what you send comes in under the licence of the path it touches. There is no contributor
licence agreement. Fonts keep their own licence (OFL-1.1), and `uvx reuse lint` checks every path in CI, so a new
top-level path needs a block in `REUSE.toml`.

Fixes to streets, buildings or places belong in [OpenStreetMap](https://www.openstreetmap.org) itself: the game
rebuilds Cuba from it, and every map that uses OSM gets the fix.

## A contribution or a mod?

What goes into this repository belongs to the common game, and every player gets it for free. What you ship
separately, needing the base game to run, is a mod: it stays yours, and `MODS.md` says how you may sell it. The name
and the logo are covered by `TRADEMARKS.md`.

## Sign off every commit

Every commit carries a `Signed-off-by` line with your name and email, which `git commit -s` adds. By signing off you
state the Developer Certificate of Origin below: that you wrote the change or have the right to send it under the
licence of the paths it touches. CI rejects a pull request with a commit that is not signed off.

```text
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.


Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```

## En español

Contravía tiene cuatro capas con su licencia cada una: el código (GPLv3 o posterior), los datos del mapa (ODbL), el
arte, la música, la historia y la documentación (CC BY-NC-SA 4.0) y los extras de pago (mods y assets, nunca mecánicas ni contenido esencial), que no están en este
repositorio. Lo que envías entra con la licencia de la ruta que toca, sin firmar ningún acuerdo aparte. Los arreglos
de calles y edificios van a OpenStreetMap. Lo que entra al repositorio es del juego común y gratis para todos; lo que
distribuyes aparte es un mod, sigue siendo tuyo y `MODS.md` dice cómo puedes venderlo. Firma cada commit con
`git commit -s`.
