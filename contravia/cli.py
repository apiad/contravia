"""The `contravia` command: prepare the city data, play, or run the benchmark."""

import argparse
import os


def _xz(text: str) -> tuple[float, float]:
    x, z = text.split(",")
    return float(x), float(z)


def _city() -> str:
    """The island to play: a checkout's data/cuba, else the first run's, built now in its window if missing."""
    import subprocess
    import sys
    from pathlib import Path

    from contravia.firstrun import built
    from contravia.ui import ROOT

    if Path("data/cuba/index.npz").exists() or Path("data/cuba/la-habana.npz").exists():
        return "data/cuba"
    if city := built():
        return str(city)
    # its own process: the build window's engine is done with before the game's opens
    code = subprocess.call([sys.executable, "-m", "contravia.cli", "build", "--window",
                            "--source", str(ROOT / "data" / "cuba-src")])
    if code or not (city := built()):
        raise SystemExit(code or 1)
    return str(city)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="contravia", description="Drive a procedural, real-size Cuba.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare", help="build each province of a Cuba OSM extract into data/cuba/<province>.npz")
    p.add_argument("pbf")
    p.add_argument("--out", default="data/cuba")
    p.add_argument("--dem", help="folder for the Copernicus GLO-30 tiles, downloaded there if missing; without it the island is flat")
    p.add_argument("--province", action="append", help="build only this province (name or slug); repeatable")
    p.add_argument("--worldcover", help="folder for the ESA WorldCover tiles, downloaded there if missing: blocks the "
                                        "satellite sees built get filler houses, not parks")
    p.add_argument("--resume", action="store_true", help="leave the provinces already built in --out as they are")
    p.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2),
                   help="provinces built at once, each in its own process of about 2.5 GB (half the cores by default)")

    p = sub.add_parser("source", help="pack a Cuba OSM extract and the GLO-30 tiles into the compact source the game ships")
    p.add_argument("pbf")
    p.add_argument("--dem", required=True, help="folder with the Copernicus GLO-30 tiles")
    p.add_argument("--worldcover", required=True, help="folder for the ESA WorldCover tiles, downloaded there if missing")
    p.add_argument("--out", default="data/cuba-src", help="folder for one <province>.src each")
    p.add_argument("--province", action="append", help="pack only this province (name or slug); repeatable")

    p = sub.add_parser("coast", help="write the map's coast (OSM natural=coastline) of an extract into a built island or the sources")
    p.add_argument("pbf")
    p.add_argument("--out", default="data/cuba", help="the built island (its index is rewritten on the next load) or data/cuba-src")

    p = sub.add_parser("shore", help="write what the shore is made of (OSM beaches, cliffs and wetlands near the coast) "
                       "into a built island or the sources, beside coast.src")
    p.add_argument("pbf")
    p.add_argument("--out", default="data/cuba", help="the folder holding coast.src; shore.src is written beside it")

    p = sub.add_parser("cover", help="write what covers the land (ESA WorldCover 2021) over the coast into a built island or the sources")
    p.add_argument("--out", default="data/cuba", help="the folder holding coast.src; cover.src is written beside it")
    p.add_argument("--cache", default="~/.cache/worldcover", help="where the tiles are downloaded; each is deleted once read")

    p = sub.add_parser("build", help="build the city from the compact source into the player's home, as the first run does")
    p.add_argument("--source", default="data/cuba-src", help="folder of the province sources")
    p.add_argument("--home", help="where to build; the per-user data folder by default")
    p.add_argument("--window", action="store_true", help="show the build screen instead of printing the progress")
    p.add_argument("--shot", help="with --window: render offscreen and save SHOT-<step>.png as the build goes")

    p = sub.add_parser("streets", help="rebuild the street surface of a prepared data file in place")
    p.add_argument("--data", default="data/cuba/la-habana.npz")

    p = sub.add_parser("walks", help="rebuild the pavements and building platforms of a prepared data file in place")
    p.add_argument("--data", default="data/cuba/la-habana.npz")

    p = sub.add_parser("play", help="drive the city")
    p.add_argument("--data", help="the built city; data/cuba in a checkout, else the one the first run builds")
    p.add_argument("--at", type=_xz, default=(0.0, 0.0), help="start point as X,Z in local metres")
    p.add_argument("--shot", help="render offscreen, save this PNG and exit")
    p.add_argument("--frames", type=int, default=120, help="frames to run before --shot is saved")
    p.add_argument("--traffic", help="JSON file of traffic settings to change (see contravia/rules/traffic.py); turns traffic on")
    p.add_argument("--with-traffic", action="store_true", help="turn the traffic on; it is off by default")
    p.add_argument("--time", help='start the clock at "17:30", "Sun 08:00" or "2026-09-12 17:30" (10:00 by default); "now" is the machine\'s clock')
    p.add_argument("--sky", help="JSON file of sky settings to change (see contravia/weather.py, SkyConfig)")
    p.add_argument("--warp", type=float, default=1.0, help="run the clock this many times faster; for tests")
    p.add_argument("--height", type=float, help="with --screen map: the map camera's height over --at, in metres")
    p.add_argument("--yaw", type=float, default=0.0, help="with --screen map: the map's turn from north, in degrees")
    p.add_argument("--place", help="a collectible place's id (contravia/places.json): start on the street nearest it; "
                   "--screen aim aims at it and --screen card photographs it")
    p.add_argument("--screen", choices=["main", "free", "options", "pause", "photo", "map", "garage", "design", "drive",
                                        "aim", "card", "album", "province"],
                   help="open on this menu screen, or drive at once; main for a window, drive for --shot")
    p.add_argument("--no-pipeline", dest="pipeline", action="store_false",
                   help="run the systems and the render one after the other instead of overlapping them")

    p = sub.add_parser("bench", help="drive the fixed route and record frame times")
    p.add_argument("--data", default="data/cuba")
    p.add_argument("--time", default="Tue 08:00", help='traffic time for the run, as "Tue 08:00"')

    p = sub.add_parser("flowmap", help="draw the flow layer over the province at a time, as a PNG")
    p.add_argument("out")
    p.add_argument("--data", default="data/cuba/la-habana.npz")
    p.add_argument("--time", default="Tue 08:00", help='as "Sun 08:00"')
    p.add_argument("--traffic", help="JSON file of traffic settings to change")
    p.add_argument("--size", type=int, default=2000, help="width in pixels")
    p.add_argument("--box", help="x0,z0,x1,z1 in local metres; the whole province by default")

    p = sub.add_parser("climate", help="build the star, Milky Way and weather tables in contravia/sky_data (needs network)")
    p.add_argument("--cache", default="~/.cache/havana", help="where the downloads are kept")
    p.add_argument("--out", help="folder for the tables; contravia/sky_data by default")

    p = sub.add_parser("skymap", help="draw a year of the weather chain as a PNG: days down, hours across")
    p.add_argument("out")
    p.add_argument("--year", type=int, default=2026)
    p.add_argument("--sky", help="JSON file of sky settings to change")

    p = sub.add_parser("fleet-sheet", help="render a review sheet per fleet kind: a row per variant in four views")
    p.add_argument("outdir")
    p.add_argument("--kind", help="only this kind of vehicle (car, bus, truck, moto, ...)")

    p = sub.add_parser("dev", help="play under hot reload: a saved file in contravia/ rebuilds the game in the same "
                                   "window, with the car where it was; takes the options of play")

    p = sub.add_parser("building-tags", help="add what OSM says about each building (kind, levels, height, use, "
                                             "municipio) to a data file prepared before")
    p.add_argument("pbf")
    p.add_argument("--data", default="data/cuba/la-habana.npz")

    p = sub.add_parser("walls-check", help="count footprints, filler boxes and landmark walls inside a road; "
                                           "all must be 0")
    p.add_argument("--data", default="data/cuba/la-habana.npz")

    p = sub.add_parser("timelapse", help="render a day of sky over the Capitolio as a video")
    p.add_argument("out")
    p.add_argument("--data", default="data/cuba")
    p.add_argument("--day", help="YYYY-MM-DD; by default the best day of the 2026 rainy season")
    p.add_argument("--start", default="04:30")
    p.add_argument("--minutes", type=int, default=1440, help="game minutes to cover")
    p.add_argument("--step", type=float, default=1.0, help="game minutes per frame")
    p.add_argument("--sky", help="JSON file of sky settings to change")

    args, extra = parser.parse_known_args(argv)
    if extra and args.command != "dev":
        parser.error(f"unrecognized arguments: {' '.join(extra)}")
    if args.command == "prepare":
        from contravia.prepare import main as prepare

        prepare(args.pbf, args.out, dem=args.dem, provinces=args.province, resume=args.resume, jobs=args.jobs,
                worldcover=args.worldcover)
    elif args.command == "source":
        from contravia.source import main as source

        source(args.pbf, args.dem, args.out, args.province, args.worldcover)
    elif args.command == "coast":
        import json
        from pathlib import Path

        from contravia import coast

        land, report = coast.read(args.pbf)
        report["bytes"] = coast.save(land, Path(args.out) / coast.NAME)
        print(json.dumps(report))
    elif args.command == "shore":
        import json
        from pathlib import Path

        from contravia import coast, shore

        features, report = shore.read(args.pbf, coast.load(Path(args.out) / coast.NAME))
        report["bytes"] = shore.save(features, Path(args.out) / shore.NAME)
        print(json.dumps(report))
    elif args.command == "cover":
        import json
        from pathlib import Path

        from contravia import coast, cover

        grid, report = cover.build(coast.load(Path(args.out) / coast.NAME), os.path.expanduser(args.cache))
        report["bytes"] = cover.save(grid, Path(args.out) / cover.NAME)
        print(json.dumps(report))
    elif args.command == "build":
        if args.window or args.shot:
            from contravia.buildscreen import main as window

            code = window(args.source, args.home, shot=args.shot)
            if code:  # stopped mid-step: the pool's threads would hold the exit until their step ends
                import sys

                sys.stdout.flush()
                os._exit(code)
            raise SystemExit(0)
        from contravia.firstrun import main as build

        build(args.source, args.home)
    elif args.command == "streets":
        from contravia.streets import main as streets

        streets(args.data)
    elif args.command == "walks":
        from contravia.walks import main as walks

        walks(args.data)
    elif args.command == "play":
        from contravia.game import main as play

        play(args.data or _city(), at=args.at, shot=args.shot, frames=args.frames, traffic=args.traffic, when=args.time,
             warp=args.warp, sky=args.sky, pipelined=args.pipeline,
             with_traffic=args.with_traffic, screen=args.screen, height=args.height, yaw=args.yaw,
             place=args.place)
    elif args.command == "dev":
        from manifoldx.cli import main as manifoldx

        manifoldx(["run", "-m", "contravia.cli", "--", "play", *extra])
    elif args.command == "building-tags":
        from contravia.buildingtags import main as building_tags

        building_tags(args.pbf, args.data)
    elif args.command == "walls-check":
        import sys

        from contravia.walls_check import main as walls_check

        sys.exit(walls_check(args.data))
    elif args.command == "climate":

        from contravia.climate import main as climate

        climate(os.path.expanduser(args.cache), args.out)
    elif args.command == "skymap":
        from contravia.skymap import main as skymap

        skymap(args.out, args.year, sky=args.sky)
    elif args.command == "fleet-sheet":
        from contravia.fleetsheet import main as fleet_sheet

        fleet_sheet(args.outdir, kind=args.kind)
    elif args.command == "timelapse":
        from contravia.timelapse import main as timelapse

        timelapse(args.data, args.out, day=args.day, start=args.start, minutes=args.minutes, step=args.step,
                  sky_file=args.sky)
    elif args.command == "flowmap":
        from contravia.flowmap import main as flowmap

        box = tuple(float(v) for v in args.box.split(",")) if args.box else None
        flowmap(args.data, args.out, args.time, traffic=args.traffic, size=args.size, box=box)
    else:
        from contravia.bench import main as bench

        bench(args.data, when=args.time)


if __name__ == "__main__":
    main()
