"""Preprocess data/raw -> data/processed.

Two jobs, both driven entirely by params.yaml:

1. Resize every image to 224 x 224 RGB (params.yaml -> data.image_size).
   We resize the shorter side to 224 and take a centre crop, which preserves
   the aspect ratio instead of squashing the animal.

2. Stratified 80 / 10 / val / test split (params.yaml -> data.split) with a
   fixed seed, so the split is deterministic and reproducible across machines.

Note: data *augmentation* is deliberately NOT done here. Augmentation is a
random, per-epoch transform applied to the training set at training time
(src/data/dataset.py); baking it into files on disk would fix the randomness
and pollute the validation/test sets.

Output layout:
    data/processed/train/{cat,dog}/...
    data/processed/val/{cat,dog}/...
    data/processed/test/{cat,dog}/...
    data/processed/split_summary.json

Run:  python src/data/preprocess.py
"""

from __future__ import annotations

import json
import random
import shutil
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import load_params, resolve  # noqa: E402

SPLITS = ("train", "val", "test")


def resize_center_crop(image: Image.Image, size: int) -> Image.Image:
    """Resize the short side to `size`, then centre-crop to size x size."""
    image = image.convert("RGB")
    width, height = image.size
    scale = size / min(width, height)
    new_size = (max(size, round(width * scale)), max(size, round(height * scale)))
    image = image.resize(new_size, Image.BILINEAR)

    left = (image.width - size) // 2
    top = (image.height - size) // 2
    return image.crop((left, top, left + size, top + size))


def split_indices(n: int, ratios: dict, seed: int) -> dict[str, list[int]]:
    """Shuffle 0..n-1 with `seed` and cut it into train/val/test index lists."""
    order = list(range(n))
    random.Random(seed).shuffle(order)

    n_train = int(n * ratios["train"])
    n_val = int(n * ratios["val"])
    return {
        "train": order[:n_train],
        "val": order[n_train:n_train + n_val],
        "test": order[n_train + n_val:],       # remainder -> test, nothing is lost
    }


def main() -> None:
    params = load_params()
    data_cfg = params["data"]
    classes = data_cfg["classes"]
    size = int(data_cfg["image_size"])
    seed = int(data_cfg["seed"])
    ratios = data_cfg["split"]

    raw_dir = resolve(data_cfg["raw_dir"])
    processed_dir = resolve(data_cfg["processed_dir"])

    if not raw_dir.is_dir():
        raise SystemExit(f"{raw_dir} not found. Run src/data/download_data.py first.")

    # Rebuild from scratch so a re-run can never leave stale files behind.
    for split in SPLITS:
        shutil.rmtree(processed_dir / split, ignore_errors=True)

    summary: dict = {
        "image_size": [size, size, data_cfg["channels"]],
        "seed": seed,
        "ratios": ratios,
        "counts": {split: {} for split in SPLITS},
    }

    for class_name in classes:
        files = sorted((raw_dir / class_name).glob("*.jpg"))
        if not files:
            raise SystemExit(f"No images found in {raw_dir / class_name}")

        # Stratified: the split is computed per class, so every split keeps
        # the same cat:dog balance as the full dataset.
        indices = split_indices(len(files), ratios, seed)

        for split in SPLITS:
            out_dir = processed_dir / split / class_name
            out_dir.mkdir(parents=True, exist_ok=True)

            for i in indices[split]:
                src_path = files[i]
                with Image.open(src_path) as image:
                    resize_center_crop(image, size).save(
                        out_dir / src_path.name, "JPEG", quality=95
                    )

            summary["counts"][split][class_name] = len(indices[split])
            print(f"{split:>5} / {class_name:<3}: {len(indices[split])} images "
                  f"-> {out_dir.relative_to(processed_dir.parent.parent)}")

    summary["totals"] = {s: sum(summary["counts"][s].values()) for s in SPLITS}
    summary["total_images"] = sum(summary["totals"].values())

    with open(processed_dir / "split_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print("\nTotals:", summary["totals"], "= ", summary["total_images"], "images")
    print(f"Summary written to {processed_dir / 'split_summary.json'}")


if __name__ == "__main__":
    main()
