"""The `havana` command: prepare the city data, play, or run the benchmark."""

import argparse


def _xz(text: str) -> tuple[float, float]:
    x, z = text.split(",")
    return float(x), float(z)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="havana", description="Drive a procedural, real-size Havana.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare", help="turn a Cuba OSM extract into data/havana.npz")
    p.add_argument("pbf")
    p.add_argument("--out", default="data/havana.npz")
    p.add_argument("--dem", help="folder with the Copernicus GLO-30 tiles; without it the city is flat")

    p = sub.add_parser("tiles", help="turn a Cuba OSM extract into the island's quadtree tiles")
    p.add_argument("pbf")
    p.add_argument("--out", default="data/cuba")
    p.add_argument("--split", type=float, default=2000.0, help="a tile splits while it weighs more than this")

    p = sub.add_parser("play", help="drive the city")
    p.add_argument("--data", default="data/havana.npz")
    p.add_argument("--at", type=_xz, default=(0.0, 0.0), help="start point as X,Z in local metres")
    p.add_argument("--shot", help="render offscreen, save this PNG and exit")
    p.add_argument("--frames", type=int, default=120, help="frames to run before --shot is saved")
    p.add_argument("--traffic", help="JSON file of traffic settings to change (see havana/rules/traffic.py)")
    p.add_argument("--time", help='pin the clock, as "Sun 08:00" or "2026-09-12 17:30"; play uses the machine\'s clock')
    p.add_argument("--sky", help="JSON file of sky settings to change (see havana/weather.py, SkyConfig)")
    p.add_argument("--warp", type=float, default=1.0, help="run the clock this many times faster; for tests")

    p = sub.add_parser("bench", help="drive the fixed route and record frame times")
    p.add_argument("--data", default="data/havana.npz")
    p.add_argument("--time", default="Tue 08:00", help='traffic time for the run, as "Tue 08:00"')

    p = sub.add_parser("flowmap", help="draw the flow layer over the province at a time, as a PNG")
    p.add_argument("out")
    p.add_argument("--data", default="data/havana.npz")
    p.add_argument("--time", default="Tue 08:00", help='as "Sun 08:00"')
    p.add_argument("--traffic", help="JSON file of traffic settings to change")
    p.add_argument("--size", type=int, default=2000, help="width in pixels")
    p.add_argument("--box", help="x0,z0,x1,z1 in local metres; the whole province by default")

    p = sub.add_parser("climate", help="build the star, Milky Way and weather tables in havana/sky_data (needs network)")
    p.add_argument("--cache", default="~/.cache/havana", help="where the downloads are kept")
    p.add_argument("--out", help="folder for the tables; havana/sky_data by default")

    p = sub.add_parser("skymap", help="draw a year of the weather chain as a PNG: days down, hours across")
    p.add_argument("out")
    p.add_argument("--year", type=int, default=2026)
    p.add_argument("--sky", help="JSON file of sky settings to change")

    p = sub.add_parser("timelapse", help="render a day of sky over the Capitolio as a video")
    p.add_argument("out")
    p.add_argument("--data", default="data/havana.npz")
    p.add_argument("--day", help="YYYY-MM-DD; by default the best day of the 2026 rainy season")
    p.add_argument("--start", default="04:30")
    p.add_argument("--minutes", type=int, default=1440, help="game minutes to cover")
    p.add_argument("--step", type=float, default=1.0, help="game minutes per frame")
    p.add_argument("--sky", help="JSON file of sky settings to change")

    args = parser.parse_args(argv)
    if args.command == "prepare":
        from havana.prepare import main as prepare

        prepare(args.pbf, args.out, dem=args.dem)
    elif args.command == "tiles":
        from havana.tiles_prepare import main as tiles

        tiles(args.pbf, args.out, args.split)
    elif args.command == "play":
        from havana.game import main as play

        play(args.data, at=args.at, shot=args.shot, frames=args.frames, traffic=args.traffic, when=args.time,
             warp=args.warp, sky=args.sky)
    elif args.command == "climate":
        import os

        from havana.climate import main as climate

        climate(os.path.expanduser(args.cache), args.out)
    elif args.command == "skymap":
        from havana.skymap import main as skymap

        skymap(args.out, args.year, sky=args.sky)
    elif args.command == "timelapse":
        from havana.timelapse import main as timelapse

        timelapse(args.data, args.out, day=args.day, start=args.start, minutes=args.minutes, step=args.step,
                  sky_file=args.sky)
    elif args.command == "flowmap":
        from havana.flowmap import main as flowmap

        box = tuple(float(v) for v in args.box.split(",")) if args.box else None
        flowmap(args.data, args.out, args.time, traffic=args.traffic, size=args.size, box=box)
    else:
        from havana.bench import main as bench

        bench(args.data, when=args.time)


if __name__ == "__main__":
    main()
