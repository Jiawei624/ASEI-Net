"""Download the original TAB archive; extraction is a separate explicit operation."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("dataset/TAB_dataset.zip"))
    args = parser.parse_args()
    import gdown
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise SystemExit(f"Already exists: {args.output}. Choose another --output to download again.")
    result = gdown.download(id="1V5BAHWBKU8uih3hE1R7WdF6_crZlIbQT", output=str(args.output), quiet=False)
    if result is None:
        raise SystemExit("Download failed. Use the official TAB Google Drive link in README.md.")


if __name__ == "__main__":
    main()
