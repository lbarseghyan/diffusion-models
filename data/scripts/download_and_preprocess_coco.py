#!/usr/bin/env python3
"""
download_and_preprocess_coco.py

Downloads and preprocesses the following COCO datasets:
 1. COCO minitrain (25K images) via Kaggle
 2. COCO 2017 validation set
 3. COCO 2017 test set

Raw downloads are stored under `data/raw/coco/` and preserved.
Processed images (target) and captions (condition) are output under `data/coco/`:
  - train/target, train/captions
  - val/target, val/captions
  - test/target

Usage:
    cd path/to/your-repo/data/scripts
    python3 download_and_preprocess_coco.py
"""

import os
import json
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path


def download_kaggle(dataset: str, target_dir: Path):
    """
    Download and unzip a Kaggle dataset via the Kaggle CLI.
    """
    print(f"Downloading Kaggle dataset {dataset} into {target_dir} …")
    subprocess.run([
        "kaggle", "datasets", "download", "-d", dataset,
        "-p", str(target_dir), "--unzip"
    ], check=True)


def download_and_extract_zip(url: str, zip_path: Path, extract_to: Path):
    """
    Download a ZIP file and extract it.
    """
    print(f"Downloading {url} to {zip_path} …")
    urllib.request.urlretrieve(url, zip_path)
    print(f"Extracting {zip_path} to {extract_to} …")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(path=extract_to)


def preprocess_split(
    name: str,
    images_dir: Path,
    annotation_json: Path or None,
    processed_images: Path,
    processed_captions: Path = None
):
    """
    Copy images and, if annotation_json is provided, write one caption per image.
    """
    print(f"Preprocessing split '{name}' …")
    processed_images.mkdir(parents=True, exist_ok=True)
    if annotation_json and processed_captions:
        processed_captions.mkdir(parents=True, exist_ok=True)
        with open(annotation_json, 'r') as f:
            coco = json.load(f)
        file_to_id = {img['file_name']: img['id'] for img in coco['images']}
        captions_map = {}
        for ann in coco['annotations']:
            iid = ann['image_id']
            captions_map.setdefault(iid, []).append(ann['caption'])

    # Iterate images
    for img_file in images_dir.glob('*.jpg'):
        fname = img_file.name
        # Copy image
        shutil.copy(img_file, processed_images / fname)

        # Write caption if available
        if annotation_json and processed_captions:
            img_id = file_to_id.get(fname)
            if img_id and img_id in captions_map:
                cap = captions_map[img_id][0]
                txt_path = processed_captions / (Path(fname).stem + '.txt')
                with open(txt_path, 'w') as tf:
                    tf.write(cap)


def main():
    this_file = Path(__file__).resolve()
    project_root = this_file.parents[2]

    raw_base = project_root / 'data' / 'raw' / 'coco'
    proc_base = project_root / 'data' / 'coco'

    # 1) COCO minitrain → train
    raw_train = raw_base / 'coco_minitrain'
    if not raw_train.exists() or not any(raw_train.iterdir()):
        raw_train.mkdir(parents=True, exist_ok=True)
        download_kaggle('trungit/coco25k', raw_train)
    else:
        print(f"COCO minitrain already at {raw_train}, skipping.")

    # find annotation JSON in raw_train
    jsons = list(raw_train.glob('*.json'))
    annot_train = jsons[0] if jsons else None
    imgs_train = raw_train / 'train2017' if (raw_train / 'train2017').exists() else raw_train
    preprocess_split(
        'train', imgs_train,
        annot_train,
        proc_base / 'train' / 'target',
        proc_base / 'train' / 'condition'
    )

    # 2) COCO val2017 → val
    raw_val = raw_base / 'val2017'
    val_zip = raw_base / 'val2017.zip'
    if not raw_val.exists():
        raw_base.mkdir(parents=True, exist_ok=True)
        download_and_extract_zip(
            'http://images.cocodataset.org/zips/val2017.zip',
            val_zip,
            raw_base
        )
    else:
        print(f"COCO val2017 already at {raw_val}, skipping.")
    # annotations
    raw_ann = raw_base / 'annotations'
    ann_zip = raw_ann / 'annotations_trainval2017.zip'
    if not (raw_ann / 'annotations').exists():
        raw_ann.mkdir(parents=True, exist_ok=True)
        download_and_extract_zip(
            'http://images.cocodataset.org/annotations/annotations_trainval2017.zip',
            ann_zip,
            raw_ann
        )
    ann_json_val = raw_ann / 'annotations' / 'captions_val2017.json'
    preprocess_split(
        'val',
        raw_val,
        ann_json_val,
        proc_base / 'val' / 'target',
        proc_base / 'val' / 'condition'
    )

    # 3) COCO test2017 → test
    raw_test = raw_base / 'test2017'
    test_zip = raw_base / 'test2017.zip'
    if not raw_test.exists():
        raw_base.mkdir(parents=True, exist_ok=True)
        download_and_extract_zip(
            'http://images.cocodataset.org/zips/test2017.zip',
            test_zip,
            raw_base
        )
    else:
        print(f"COCO test2017 already at {raw_test}, skipping.")
    preprocess_split(
        'test',
        raw_test,
        None,
        proc_base / 'test' / 'target'
    )

    print("Finished downloading and preprocessing all COCO splits.")

if __name__ == '__main__':
    main()
