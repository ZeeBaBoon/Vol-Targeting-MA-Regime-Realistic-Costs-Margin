"""`uv run voltarget [run|figures|all] [--refresh]`"""

import argparse


def main():
    p = argparse.ArgumentParser(prog="voltarget")
    p.add_argument("step", choices=["run", "figures", "all"], nargs="?", default="all")
    p.add_argument("--refresh", action="store_true", help="re-download prices from Yahoo Finance")
    a = p.parse_args()
    if a.refresh:
        from .data import load
        load(refresh=True)
    if a.step in ("run", "all"):
        from .experiments import report_text, run
        print(report_text(run()))
    if a.step in ("figures", "all"):
        from .figures import run as figs
        figs()


if __name__ == "__main__":
    main()
