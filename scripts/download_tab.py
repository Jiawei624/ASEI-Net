"""Download the original TAB archive; extraction is a separate explicit operation."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asei.config import DEFAULT_CONFIG, load_config, project_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "--config-path", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output", type=Path, help="Override data.root/data.archive")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.output:
        target = project_path(args.output)
    else:
        archive = Path(cfg["data"].get("archive", "TAB_dataset.zip")).expanduser()
        target = archive if archive.is_absolute() else Path(cfg["data"]["root"]) / archive
    import gdown
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise SystemExit(f"Already exists: {target}. Choose another --output to download again.")
    result = gdown.download(id="1V5BAHWBKU8uih3hE1R7WdF6_crZlIbQT", output=str(target), quiet=False)
    if result is None:
        raise SystemExit("Download failed. Use the official TAB Google Drive link in README.md.")


if __name__ == "__main__":
    main()
