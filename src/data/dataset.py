"""PyTorch Datasets / DataLoaders for the Cats vs Dogs pipeline.

Key rule enforced here: **augmentation is applied to the training split only.**
Validation and test images get resize/normalise only, so their scores are an
honest estimate of real-world performance.

Class indices come from torchvision's ImageFolder, which sorts folder names
alphabetically -> cat = 0, dog = 1. That matches data.classes in params.yaml,
and build_dataloaders() asserts it rather than trusting it.

Run:  python src/data/dataset.py     (sanity check + sample augmentation grid)
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import load_params, resolve  # noqa: E402


def build_transforms(params: dict, train: bool) -> transforms.Compose:
    """Build the transform pipeline for one split.

    train=True  -> random augmentation (new every epoch) + normalise
    train=False -> deterministic resize + normalise only
    """
    data_cfg = params["data"]
    size = int(data_cfg["image_size"])
    norm = transforms.Normalize(
        mean=data_cfg["normalize"]["mean"], std=data_cfg["normalize"]["std"]
    )

    if not train:
        # Images on disk are already 224x224, but Resize keeps this pipeline
        # correct for any image handed to it (e.g. an upload to the API).
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            norm,
        ])

    aug = params["augmentation"]
    steps: list = [
        transforms.RandomResizedCrop(
            size, scale=tuple(aug["random_resized_crop_scale"]), antialias=True
        )
    ]
    if aug.get("horizontal_flip", False):
        steps.append(transforms.RandomHorizontalFlip(p=0.5))
    if aug.get("rotation_degrees", 0):
        steps.append(transforms.RandomRotation(aug["rotation_degrees"]))
    if aug.get("color_jitter", 0):
        jitter = float(aug["color_jitter"])
        steps.append(
            transforms.ColorJitter(
                brightness=jitter, contrast=jitter, saturation=jitter
            )
        )
    steps += [transforms.ToTensor(), norm]
    return transforms.Compose(steps)


def build_datasets(params: dict) -> dict[str, ImageFolder]:
    processed_dir = resolve(params["data"]["processed_dir"])
    datasets = {}
    for split in ("train", "val", "test"):
        split_dir = processed_dir / split
        if not split_dir.is_dir():
            raise SystemExit(
                f"{split_dir} not found. Run src/data/preprocess.py first."
            )
        datasets[split] = ImageFolder(
            split_dir, transform=build_transforms(params, train=(split == "train"))
        )

    # Fail loudly if the label order ever drifts from params.yaml.
    expected = list(params["data"]["classes"])
    actual = datasets["train"].classes
    if actual != expected:
        raise SystemExit(
            f"Class order mismatch: ImageFolder found {actual}, "
            f"params.yaml declares {expected}"
        )
    return datasets


def build_dataloaders(params: dict) -> tuple[dict[str, DataLoader], list[str]]:
    """Return {'train','val','test'} -> DataLoader, plus the class name list."""
    train_cfg = params["train"]
    datasets = build_datasets(params)

    # Seed the shuffling so a re-run of training sees the same batch order.
    generator = torch.Generator().manual_seed(int(train_cfg["seed"]))

    loaders = {
        split: DataLoader(
            dataset,
            batch_size=int(train_cfg["batch_size"]),
            shuffle=(split == "train"),
            num_workers=int(train_cfg["num_workers"]),
            generator=generator if split == "train" else None,
            pin_memory=torch.cuda.is_available(),
        )
        for split, dataset in datasets.items()
    }
    return loaders, datasets["train"].classes


def _sanity_check() -> None:
    params = load_params()
    loaders, classes = build_dataloaders(params)

    print(f"classes            : {classes}  (index 0 -> {classes[0]})")
    for split, loader in loaders.items():
        print(f"{split:>5}: {len(loader.dataset):>4} images, "
              f"{len(loader):>3} batches, augmented={split == 'train'}")

    images, labels = next(iter(loaders["train"]))
    print(f"\nbatch tensor shape : {tuple(images.shape)}  dtype={images.dtype}")
    print(f"batch label shape  : {tuple(labels.shape)}  values={sorted(set(labels.tolist()))}")
    print(f"normalised range   : min={images.min():.3f} max={images.max():.3f} "
          f"mean={images.mean():.3f}")

    # Proof that augmentation is random: the same image twice, different result.
    dataset = loaders["train"].dataset
    a, _ = dataset[0]
    b, _ = dataset[0]
    print(f"same index twice differs (augmentation is live): "
          f"{not torch.allclose(a, b)}")

    # Save a visual grid for the report/demo.
    from torchvision.utils import save_image
    mean = torch.tensor(params["data"]["normalize"]["mean"]).view(3, 1, 1)
    std = torch.tensor(params["data"]["normalize"]["std"]).view(3, 1, 1)
    grid_path = resolve(params["model"]["artifact_dir"]) / "sample_augmented_batch.png"
    grid_path.parent.mkdir(parents=True, exist_ok=True)
    save_image((images[:16] * std + mean).clamp(0, 1), grid_path, nrow=4)
    print(f"sample grid written: {grid_path}")


if __name__ == "__main__":
    _sanity_check()
