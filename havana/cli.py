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

    p = sub.add_parser("play", help="drive the city")
    p.add_argument("--data", default="data/havana.npz")
    p.add_argument("--at", type=_xz, default=(0.0, 0.0), help="start point as X,Z in local metres")
    p.add_argument("--shot", help="render offscreen, save this PNG and exit")
    p.add_argument("--frames", type=int, default=120, help="frames to run before --shot is saved")

    p = sub.add_parser("bench", help="drive the fixed route and record frame times")
    p.add_argument("--data", default="data/havana.npz")

    args = parser.parse_args(argv)
    if args.command == "prepare":
        from havana.prepare import main as prepare

        prepare(args.pbf, args.out)
    else:
        print(f"havana {args.command}: not built yet")


if __name__ == "__main__":
    main()
