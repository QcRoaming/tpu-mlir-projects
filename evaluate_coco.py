#!/usr/bin/env python3
"""Evaluate local COCO detection results with a fixed image set and optional baseline."""

import argparse
import json
import math
from pathlib import Path


def validate_detections(detections, image_ids, category_ids):
    if not isinstance(detections, list):
        raise ValueError("COCO detections must be a JSON array")
    for detection in detections:
        if not isinstance(detection, dict):
            raise ValueError("Each detection must be a JSON object")
        if detection.get("image_id") not in image_ids:
            raise ValueError(f"Unknown image_id: {detection.get('image_id')}")
        if detection.get("category_id") not in category_ids:
            raise ValueError(f"Unknown category_id: {detection.get('category_id')}")
        box = detection.get("bbox", [])
        if not isinstance(box, (list, tuple)) or len(box) != 4 or not all(
                isinstance(x, (int, float)) and math.isfinite(x) for x in box):
            raise ValueError("Each bbox must contain four finite xywh values")
        if box[2] < 0 or box[3] < 0:
            raise ValueError("COCO bbox width and height must be nonnegative")
        score = detection.get("score")
        if not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Detection scores must be finite probabilities")


def score_coco(ground_truth, detections, image_ids):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    validate_detections(detections, set(ground_truth.getImgIds()), set(ground_truth.getCatIds()))
    if detections:
        prediction = ground_truth.loadRes(detections)
    else:
        # COCO.loadRes indexes element zero and cannot handle an empty result list.
        prediction = COCO()
        prediction.dataset = {"images": ground_truth.dataset["images"],
                              "categories": ground_truth.dataset["categories"], "annotations": []}
        prediction.createIndex()
    evaluator = COCOeval(ground_truth, prediction, "bbox")
    evaluator.params.imgIds = sorted(image_ids)
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    names = ["AP_50_95", "AP_50", "AP_75", "AP_small", "AP_medium", "AP_large",
             "AR_1", "AR_10", "AR_100", "AR_small", "AR_medium", "AR_large"]
    return dict(zip(names, (float(value) for value in evaluator.stats)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--baseline-results", type=Path)
    parser.add_argument("--image-ids", type=Path,
                        help="JSON array of all evaluated images, "
                        "including those without detections")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from pycocotools.coco import COCO
    ground_truth = COCO(str(args.annotations.resolve()))
    image_ids = (json.loads(args.image_ids.read_text()) if args.image_ids else
                 ground_truth.getImgIds())
    if (not isinstance(image_ids, list) or not image_ids or
            not set(image_ids) <= set(ground_truth.getImgIds())):
        raise ValueError("Evaluation image IDs must be a nonempty subset of the annotation images")
    detections = json.loads(args.results.read_text())
    report = {"annotations": str(args.annotations.resolve()),
              "results": str(args.results.resolve()),
              "image_ids": sorted(set(image_ids)), "metric": "COCO bbox AP@[.50:.95]",
              "candidate": score_coco(ground_truth, detections, image_ids)}
    if args.baseline_results:
        baseline = json.loads(args.baseline_results.read_text())
        report["baseline"] = score_coco(ground_truth, baseline, image_ids)
        candidate_ap = report["candidate"]["AP_50_95"]
        baseline_ap = report["baseline"]["AP_50_95"]
        if candidate_ap >= 0 and baseline_ap >= 0:
            report["AP_drop_percentage_points"] = (baseline_ap - candidate_ap) * 100
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")


if __name__ == "__main__":
    main()
