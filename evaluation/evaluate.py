"""Evaluate the classical detector against YOLO box/polygon annotations.

Annotations are used ONLY for scoring, never as detector inputs. Run from repo root.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.detection import DetectorConfig, detect_plate


def read_annotations(path, width, height, classes=(0, 1)):
    result = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        values = np.asarray([float(v) for v in line.split()])
        if not np.isfinite(values).all() or values[0] != int(values[0]):
            raise ValueError(f"Invalid label at {path}:{line_no}")
        if int(values[0]) not in classes:
            continue
        coords = values[1:]
        if np.any(coords < 0) or np.any(coords > 1):
            raise ValueError(f"Coordinates outside [0,1] at {path}:{line_no}")
        if len(coords) == 4:
            cx, cy, w, h = coords
            if min(w, h) <= 0:
                raise ValueError("Invalid box size")
            points = np.array([[cx-w/2,cy-h/2],[cx+w/2,cy-h/2],
                               [cx+w/2,cy+h/2],[cx-w/2,cy+h/2]])
            kind = "box"
        elif len(coords) >= 6 and len(coords) % 2 == 0:
            points = coords.reshape(-1, 2)
            if len(points) > 3 and np.allclose(points[0], points[-1]):
                points = points[:-1]
            kind = "polygon"
        else:
            raise ValueError(f"Unsupported YOLO label at {path}:{line_no}")
        points = np.clip(points * [width, height], [0, 0], [width-1, height-1]).astype(np.float32)
        result.append({"class": int(values[0]), "kind": kind, "points": points})
    return result


def polygon_mask(points, shape):
    mask = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 1)
    return mask.astype(bool)


def overlap(points, truths, shape):
    if points is None or not truths:
        return 0.
    predicted = polygon_mask(points, shape)
    return max(float(np.count_nonzero(predicted & truth) / max(1, np.count_nonzero(predicted | truth)))
               for truth in truths)


def summarize(rows):
    n = len(rows)
    accepted = sum(r["selected"] is not None for r in rows)
    correct = sum(r["selected_iou"] >= .5 for r in rows)
    return {"images": n, "accepted": accepted, "correct_iou50": correct,
            "correct_iou75": sum(r["selected_iou"] >= .75 for r in rows),
            "incorrect_accepted": accepted-correct, "manual_required": n-accepted,
            "correct_per_image": round(correct/max(1,n),4),
            "precision_when_accepted": round(correct/max(1,accepted),4),
            "candidate_recall_iou50": sum(r["best_candidate_iou"] >= .5 for r in rows),
            "mean_selected_iou_including_rejections": round(float(np.mean([r["selected_iou"] for r in rows])),4) if rows else 0,
            "median_ms": round(float(np.median([r["elapsed_ms"] for r in rows])),2) if rows else 0}


def evaluate(root, manifest, split, config):
    rows = []
    for entry in manifest["entries"]:
        if split != "all" and entry["split"] != split:
            continue
        path = root / entry["image"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"Image changed since split: {path}")
        image = np.array(Image.open(path).convert("RGB"))
        labels = read_annotations(root / entry["label"], image.shape[1], image.shape[0])
        truths = [polygon_mask(label["points"], image.shape) for label in labels]
        start = perf_counter()
        detection = detect_plate(image, config)
        candidates = []
        for candidate in detection.candidates:
            candidates.append({"box": candidate.box, "corners": None if candidate.corners is None else candidate.corners.tolist(),
                               "score": candidate.score, "accepted": candidate.accepted, "reason": candidate.reason,
                               "metrics": candidate.metrics, "sources": sorted(candidate.sources),
                               "iou": overlap(candidate.corners, truths, image.shape)})
        chosen = next((i for i,c in enumerate(detection.candidates) if c is detection.selected), None)
        rows.append({"image": entry["image"], "split": entry["split"], "labels": len(labels),
                     "label_kinds": [l["kind"] for l in labels], "selected": chosen,
                     "selected_iou": candidates[chosen]["iou"] if chosen is not None else 0.,
                     "best_candidate_iou": max((c["iou"] for c in candidates), default=0.),
                     "elapsed_ms": round((perf_counter()-start)*1000,2), "reason": detection.reason,
                     "candidates": candidates})
        print(f"{len(rows)} {path.name[:20]} selected={chosen} IoU={rows[-1]['selected_iou']:.3f}", flush=True)
    return {"config": vars(config), "split": split, "summary": summarize(rows), "rows": rows,
            "metric": "max per-object raster polygon IoU (bbox annotations converted to rectangles); IoU >=0.5 localization, not OCR accuracy"}


def contact_sheet(root, report, output):
    tiles = []
    for index,row in enumerate(report["rows"]):
        im = np.array(Image.open(root/row["image"]).convert("RGB"))
        labels = read_annotations(root / row["image"].replace("/images/", "/labels/").rsplit('.',1)[0].__add__('.txt'), im.shape[1], im.shape[0])
        for label in labels:
            cv2.polylines(im,[np.rint(label['points']).astype(np.int32)],True,(0,210,80),3)
        if row['selected'] is not None:
            c=row['candidates'][row['selected']]
            cv2.polylines(im,[np.rint(c['corners']).astype(np.int32)],True,(255,60,30),3)
        tile=Image.new('RGB',(220,252),'white')
        tile.paste(Image.fromarray(im).resize((220,220)),(0,0))
        draw=ImageDraw.Draw(tile)
        draw.text((4,223),f"{index}: {Path(row['image']).name[:12]} IoU={row['selected_iou']:.2f}",fill='black')
        tiles.append(tile)
    sheet=Image.new('RGB',(220*6,252*((len(tiles)+5)//6)), '#dce4ed')
    for i,tile in enumerate(tiles):sheet.paste(tile,((i%6)*220,(i//6)*252))
    sheet.save(output)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,default=Path('data/thai_plates_v1'))
    parser.add_argument('--manifest',type=Path,default=Path('evaluation/split.json'))
    parser.add_argument('--split',choices=['tune','test','all'],default='tune')
    parser.add_argument('--config',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--sheet',type=Path)
    args=parser.parse_args()
    config=DetectorConfig(**json.loads(args.config.read_text())) if args.config else DetectorConfig()
    report=evaluate(args.data,json.loads(args.manifest.read_text()),args.split,config)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    if args.sheet:contact_sheet(args.data,report,args.sheet)
    print(json.dumps(report['summary'],indent=2))
