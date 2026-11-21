"""What the «Créditos y licencias» screen says: who made the game, where its data comes from, and its licences.

The data lines quote the Data section of README.md word for word, because those are the attributions the data's
licences ask for; tests/test_credits.py keeps the two the same. The licence lines name the four layers that
REUSE.toml maps each path to.
"""

SECTIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Contravía", ("Un juego abierto, hecho en Cuba sobre manifoldx.",)),
    ("Datos", (
        "© OpenStreetMap contributors, under the Open Database License.",
        "Copernicus GLO-30 Digital Elevation Model, © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH "
        "2014-2018, provided under COPERNICUS by the European Union and ESA.",
        "© ESA WorldCover project 2021 / Contains modified Copernicus Sentinel data (2021) processed by ESA "
        "WorldCover consortium, under CC BY 4.0.",
        "contains modified Copernicus Climate Change Service information (ERA5 hourly data 1991-2020, via "
        "Open-Meteo, under CC BY 4.0) and NOAA GHCN-Daily records.",
        "the Yale Bright Star Catalogue and NASA's Deep Star Maps 2020 (Scientific Visualization Studio), both in "
        "the public domain.",
    )),
    ("Tipografía", ("Noto Sans © The Noto Project Authors, SIL Open Font License 1.1",)),
    ("Motor", ("manifoldx, licencia MIT",)),
    ("Licencias", (
        "Código: GPLv3 o posterior",
        "Arte, música e historia: CC BY-NC-SA 4.0",
        "Datos del mapa: ODbL 1.0",
        "Mods: Licencia de Mods de Contravía (MODS.md)",
    )),
)
