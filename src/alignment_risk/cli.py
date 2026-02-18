from __future__ import annotations

import argparse

from .demo import run_demo


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AIC alignment risk template")
    sub = parser.add_subparsers(dest="command")

    demo = sub.add_parser("demo", help="run a synthetic end-to-end diagnostic")
    demo.add_argument("--output-dir", default="artifacts", help="where to save plots")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command in (None, "demo"):
        output_dir = getattr(args, "output_dir", "artifacts")
        run_demo(output_dir=output_dir)
        return

    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
