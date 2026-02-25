from __future__ import annotations

import argparse
from typing import Literal, cast

from .demo import run_demo


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AIC alignment risk template")
    sub = parser.add_subparsers(dest="command")

    demo = sub.add_parser("demo", help="run a synthetic end-to-end diagnostic")
    demo.add_argument("--output-dir", default="artifacts", help="where to save plots")
    demo.add_argument(
        "--mode",
        default="full",
        choices=["full", "lora"],
        help="analysis mode: full fine-tuning weights or LoRA adapter-only weights",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == "demo":
        output_dir = getattr(args, "output_dir", "artifacts")
        mode = cast(str, getattr(args, "mode", "full"))
        run_demo(output_dir=output_dir, mode=cast(Literal["full", "lora"], mode))
        return

    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
