"""Command line: ``heldout summarise`` then ``heldout figures``."""
from __future__ import annotations

import argparse


def main() -> None:
    p = argparse.ArgumentParser(prog="heldout", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("summarise", help="compute the curves and write results/summary.json")
    s.add_argument("--n-boot", type=int, default=2000)
    sub.add_parser("figures", help="draw docs/img from results/summary.json")
    args = p.parse_args()
    if args.command == "summarise":
        from .analysis import summarise
        out = summarise(args.n_boot)
        for name, c in (("crop", out["crop"]), ("views", out["views"]), ("runtime, gradient boosting", out["runtime"]["methods"]["gbm"]),
                        ("runtime, simulator", out["runtime"]["methods"]["sim"])):
            lo, hi = c["last_over_first_ci"]
            print(f"{name}: error in the farthest group is {c['last_over_first']:.2f} times the nearest ({lo:.2f} to {hi:.2f})")
    else:
        from .figures import make_all
        make_all()


if __name__ == "__main__":
    main()
