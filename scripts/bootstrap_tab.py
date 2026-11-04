"""Fetch only the pinned upstream source. No TAB model dependencies are installed."""
import argparse
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asei.config import ROOT, TAB_COMMIT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, default=ROOT / "third_party" / "TAB")
    args = parser.parse_args()
    if not args.path.exists():
        subprocess.run(["git", "clone", "--no-checkout", "--filter=blob:none",
                        "https://github.com/decisionintelligence/TAB.git", str(args.path)], check=True)
        subprocess.run(["git", "-C", str(args.path), "checkout", "--detach", TAB_COMMIT], check=True)
    else:
        revision = subprocess.check_output(["git", "-C", str(args.path), "rev-parse", "HEAD"], text=True).strip()
        if revision != TAB_COMMIT:
            raise SystemExit(f"Existing checkout is {revision}; use a new --path for the pinned revision.")
    print(f"TAB ready: {args.path} @ {TAB_COMMIT}")


if __name__ == "__main__":
    main()
