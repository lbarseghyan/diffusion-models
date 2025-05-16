#!/usr/bin/env python3
"""
download_cifar10.py

Downloads the CIFAR-10 dataset, extracts each image to disk, then
cleans up the raw download folder.

Usage:
    cd path/to/your-repo/data/scripts
    python3 download_and_preprocess_cifar10.py
"""

import argparse
import shutil
from pathlib import Path
from tqdm import tqdm
import torchvision
from torchvision import transforms

def download_dataset(root: Path):
    print(f"Downloading CIFAR-10 to {root} …")
    transform = transforms.ToTensor()  # we just need it to trigger download
    # train=False/test=False triggers download but we will re-load later for saving
    torchvision.datasets.CIFAR10(root=str(root), train=True, download=True, transform=transform)
    torchvision.datasets.CIFAR10(root=str(root), train=False, download=True, transform=transform)

def save_split(split: str, root: Path, out_dir: Path):
    """
    split: 'train' or 'test'
    """
    print(f"Saving {split} images to {out_dir} …")
    ds = torchvision.datasets.CIFAR10(root=str(root), train=(split=='train'),
                                      download=False, transform=None)
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, (img, _) in enumerate(tqdm(ds, desc=split)):
        # name files train_00001.png, test_00001.png, etc.
        fn = f"{split}_{idx:05d}.png"
        img.save(out_dir / fn)

def clean_up(raw_dir: Path):
    if raw_dir.exists():
        print(f"Removing raw directory {raw_dir} …")
        shutil.rmtree(raw_dir)
    else:
        print(f"No raw data directory at {raw_dir}, skipping clean up.")

def main():
    # determine project root as two levels up from this script
    this_file = Path(__file__).resolve()
    project_root = this_file.parents[2]

    raw_dir = project_root / "data" / "raw"
    cifar10_root = raw_dir / "cifar-10-batches-py"
    out_base = project_root / "data" / "cifar-10"

    # download into raw_dir
    download_dataset(raw_dir)

    # save images
    save_split("train", raw_dir, out_base / "train")
    save_split("test",  raw_dir, out_base / "test")

    # delete raw downloads
    # clean_up(raw_dir)

if __name__ == "__main__":
    main()