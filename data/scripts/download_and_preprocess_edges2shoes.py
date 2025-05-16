#!/usr/bin/env python3
"""
Downloads and preprocesses the pix2pix edges2shoes dataset.
- Downloads the TAR archive into `data/raw/edges2shoes/`  
- Extracts raw data in `data/raw/edges2shoes/`  
- Splits paired images into condition (edges) and target (shoes)

Usage:
    cd path/to/your-repo/data/scripts
    python3 download_edges2shoes.py
"""

import tarfile
import urllib.request
import os
from pathlib import Path
from PIL import Image


def download_dataset(url: str, archive_path: Path):
    print(f"Downloading dataset from {url} to {archive_path} …")
    urllib.request.urlretrieve(url, archive_path)


def extract_dataset(archive_path: Path, extract_to: Path):
    print(f"Extracting {archive_path} to {extract_to} …")
    with tarfile.open(archive_path, 'r:gz') as tar:
        tar.extractall(path=extract_to)


def prepare_images(input_folder: Path, condition_folder: Path, target_folder: Path):
    # Create output dirs
    os.makedirs(condition_folder, exist_ok=True)
    os.makedirs(target_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if not filename.lower().endswith('.jpg'):
            continue
        img_path = input_folder / filename
        with Image.open(img_path) as img:
            w, h = img.size
            # Crop left half (edges)
            left = img.crop((0, 0, w//2, h))
            # Crop right half (photo)
            right = img.crop((w//2, 0, w, h))

            base = filename.split('_')[0]
            left_fn  = f"{base}_A.jpg"
            right_fn = f"{base}_B.jpg"

            left.save(condition_folder / left_fn)
            right.save(target_folder  / right_fn)


def main():
    this_file = Path(__file__).resolve()
    project_root = this_file.parents[2]
    data_dir = project_root / 'data'

    # Raw data directory and archive
    raw_base = data_dir / 'raw' 
    raw_base.mkdir(parents=True, exist_ok=True)
    raw_archive = raw_base / 'edges2shoes.tar.gz'

    # Processed output directory
    processed_base = data_dir / 'edges2shoes'

    url = 'https://efrosgans.eecs.berkeley.edu/pix2pix/datasets/edges2shoes.tar.gz'

    # Download & extract raw data if train/val folders are missing
    if not (raw_base / 'edges2shoes' / 'train').exists() or not (raw_base / 'edges2shoes' / 'val').exists():
        if not raw_archive.exists():
            download_dataset(url, raw_archive)
        else:
            print(f"Archive {raw_archive} found; skipping download.")
        extract_dataset(raw_archive, raw_base)
    else:
        print(f"Raw dataset already extracted at {raw_base}; skipping.")

    # Preprocess splits into processed_base
    for split in ['train', 'val']:
        inp  = raw_base / 'edges2shoes' / split
        cond = processed_base / split / "condition"
        targ = processed_base / split / "target"
        print(f"Processing split '{split}' …")
        prepare_images(inp, cond, targ)

    print("Done preprocessing edges2shoes. Raw data remains under data/raw/edges2shoes.")

if __name__ == '__main__':
    main()

