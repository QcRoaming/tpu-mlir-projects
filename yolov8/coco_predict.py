"""Generate COCO detections from local images with the same runtime geometry/NMS."""

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from yolov8.backends import BmodelBackend, OnnxBackend
    from yolov8.runtime import launch_cmodel_if_needed
    from yolov8.vision import postprocess, preprocess
else:
    from .backends import BmodelBackend, OnnxBackend
    from .runtime import launch_cmodel_if_needed
    from .vision import postprocess, preprocess


COCO_CATEGORY_IDS = (
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51,
    52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77,
    78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90)


def predict(backend, images, annotations, category_ids=COCO_CATEGORY_IDS, limit=0,
            confidence=0.001, iou=0.65):
    root = Path(images).resolve()
    data = json.loads(Path(annotations).read_text(encoding="utf-8"))
    entries = data["images"][:limit] if limit else data["images"]
    if not root.is_dir() or not entries or limit < 0:
        raise ValueError("Require a local image directory and non-empty annotation image set")
    if len({entry["id"] for entry in entries}) != len(entries):
        raise ValueError("COCO image ids must be unique")
    if len(category_ids) != len(set(category_ids)) or not category_ids:
        raise ValueError("Category map must contain unique category ids in model class order")
    declared = {entry["id"] for entry in data.get("categories", [])}
    if declared and not set(category_ids).issubset(declared):
        raise ValueError("Category map does not match the annotation categories; "
                         "supply a local map")
    tensor = np.empty(backend.input_shape, np.float32)
    results, image_ids = [], []
    for entry in entries:
        path = (root / entry["file_name"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Annotation image path escapes --images directory")
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Cannot decode local COCO image: {path}")
        meta = preprocess(image, tensor)
        output = backend.infer(tensor)
        detections = postprocess(output, meta, len(category_ids), confidence, iou)
        image_ids.append(entry["id"])
        for x1, y1, x2, y2, score, label in detections:
            if x2 <= x1 or y2 <= y1:
                continue
            results.append(dict(image_id=entry["id"], category_id=category_ids[int(label)],
                                bbox=[float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                                score=float(score)))
    return results, image_ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--backend", choices=("onnx", "cmodel", "device"), default="onnx")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output-name")
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--category-map", type=Path,
                        help="JSON array: category ids in the model's class order")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--confidence", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.65)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.backend == "cmodel":
        launch_cmodel_if_needed(__file__, sys.argv[1:])
    backend = (OnnxBackend(args.model, args.output_name) if args.backend == "onnx" else
               BmodelBackend(args.model, args.backend, args.device, args.output_name))
    category_ids = (json.loads(args.category_map.read_text(encoding="utf-8"))
                    if args.category_map else COCO_CATEGORY_IDS)
    results, image_ids = predict(backend, args.images, args.annotations, category_ids,
                                args.limit, args.confidence, args.iou)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results), encoding="utf-8")
    args.output.with_suffix(".image_ids.json").write_text(json.dumps(image_ids), encoding="utf-8")
    print(f"Wrote {len(results)} detections from {len(image_ids)} images to {args.output}")


if __name__ == "__main__":
    main()
