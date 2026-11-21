# Contravía

An open-world driving game set in a real-size Havana, built with manifoldx for the desktop. The city is generated procedurally from OpenStreetMap and elevation data, with per-zone building recipes and textures, so the game ships rules and a small asset library instead of a modelled city. A few landmarks are modelled by hand from photos.

The design is being specified in `docs/`.

## Data

- Streets, buildings, places and the coast: © OpenStreetMap contributors, under the Open Database License.
- Relief: Copernicus GLO-30 Digital Elevation Model, © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018, provided under COPERNICUS by the European Union and ESA.
- Land cover on the map: © ESA WorldCover project 2021 / Contains modified Copernicus Sentinel data (2021) processed by ESA WorldCover consortium, under CC BY 4.0.
- Weather: contains modified Copernicus Climate Change Service information (ERA5 hourly data 1991-2020, via Open-Meteo, under CC BY 4.0) and NOAA GHCN-Daily records.
- Night sky: the Yale Bright Star Catalogue and NASA's Deep Star Maps 2020 (Scientific Visualization Studio), both in the public domain.

## Licence

- Code: GPL-3.0-or-later, in `LICENSE`.
- Map data: ODbL-1.0, in `LICENSES/ODbL-1.0.txt`.
- Core assets, story and docs: CC-BY-NC-SA-4.0, in `LICENSES/CC-BY-NC-SA-4.0.txt`; fonts under OFL-1.1.
- Mods may be sold under the terms of [`MODS.md`](MODS.md); the name and logo are covered by [`TRADEMARKS.md`](TRADEMARKS.md).
- `REUSE.toml` gives the licence of every path. To contribute, read [`CONTRIBUTING.md`](CONTRIBUTING.md).
