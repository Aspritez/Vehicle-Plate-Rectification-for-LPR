#!/usr/bin/env python3
"""Train the SIFT plate locator from LabelMe annotations.

Usage (from the backend folder):
    python train_locator.py            # cross-validate, then train on all labels and save
    python train_locator.py --no-cv    # skip cross-validation

Annotations: LabelMe .json files next to their images (default: ../data/raw_images),
each with a polygon labelled "license_plate".
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from app.core.plate_locator import SIFTPlateLocator

DEFAULT_DATA = Path(__file__).parent.parent / "data" / "raw_images"
MODEL_PATH = Path(__file__).parent / "app" / "models" / "plate_locator.joblib"


def load_dataset(data_dir: Path):
    names, images, polygons = [], [], []
    for json_path in sorted(data_dir.glob("*.json")):
        ann = json.loads(json_path.read_text(encoding="utf-8"))
        shapes = [s for s in ann.get("shapes", []) if s.get("label") == "license_plate"]
        image = cv2.imread(str(data_dir / ann["imagePath"]))
        if not shapes or image is None:
            print(f"skip {json_path.name}")
            continue
        names.append(json_path.stem)
        images.append(image)
        polygons.append(np.array(shapes[0]["points"], np.float32))
    return names, images, polygons


def bbox(points) -> tuple:
    p = np.asarray(points)
    return p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()


def iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def cross_validate(images, polygons, folds: int = 5) -> np.ndarray:
    n = len(images)
    scores = np.zeros(n)
    for k in range(folds):
        test = list(range(k, n, folds))
        train = [i for i in range(n) if i not in test]
        locator = SIFTPlateLocator()
        locator.fit([images[i] for i in train], [polygons[i] for i in train])
        for i in test:
            corners, _ = locator.locate(images[i])
            scores[i] = iou(bbox(corners), bbox(polygons[i])) if corners else 0.0
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=MODEL_PATH)
    parser.add_argument("--no-cv", action="store_true")
    args = parser.parse_args()

    names, images, polygons = load_dataset(args.data)
    print(f"Loaded {len(images)} labeled images from {args.data}")
    if len(images) < 10:
        sys.exit("Need at least 10 labeled images to train.")

    if not args.no_cv:
        scores = cross_validate(images, polygons)
        print(f"5-fold cross-validation: IoU>0.5 on {(scores > 0.5).sum()}/{len(scores)} "
              f"({(scores > 0.5).mean():.0%}), IoU>0.3 on {(scores > 0.3).mean():.0%}, "
              f"mean IoU {scores.mean():.2f}")

    locator = SIFTPlateLocator()
    stats = locator.fit(images, polygons)
    locator.save(args.out)
    print(f"Trained on {stats['plate_keypoints']} plate / {stats['keypoints']} total keypoints")
    print(f"Saved model to {args.out}")


if __name__ == "__main__":
    main()
