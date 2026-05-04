from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Kaggle legalraw dataset into raw_data.")
    parser.add_argument("--dataset", default="dlvin755/legalraw", help="Kaggle dataset slug.")
    parser.add_argument("--output_dir", type=Path, default=Path("raw_data/legalraw/full"))
    parser.add_argument("--force", action="store_true", help="Overwrite existing files.")
    return parser.parse_args()


def copy_file(src: Path, dst: Path, *, force: bool) -> None:
    if dst.exists() and not force:
        print(f"[skip] {dst}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"[save] {dst}")


def main() -> None:
    args = parse_args()
    try:
        import kagglehub
    except ImportError as exc:
        raise SystemExit("Please install kagglehub first: python3 -m pip install kagglehub") from exc

    cache_path = Path(kagglehub.dataset_download(args.dataset))
    expected = ["legal_corpus.json", "train.json"]
    missing = [name for name in expected if not (cache_path / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing expected files in {cache_path}: {missing}")

    for name in expected:
        copy_file(cache_path / name, args.output_dir / name, force=args.force)

    print(f"[done] dataset copied to {args.output_dir}")


if __name__ == "__main__":
    main()
