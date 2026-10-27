"""Write calibration NPZ inputs using exactly the runtime letterbox operation."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

if __package__ in {None, ""}:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from yolov8.vision import preprocess
else:
    from .vision import preprocess


def prepare(dataset, output, input_name="images", size=640, limit=100):
    dataset, output = Path(dataset).resolve(), Path(output).resolve()
    if not dataset.is_dir() or limit < 1 or size < 32:
        raise ValueError("Require an existing image directory, positive limit and size >= 32")
    images = sorted(p for p in dataset.rglob("*")
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})[:limit]
    if not images:
        raise ValueError("Calibration dataset contains no supported image files")
    output.mkdir(parents=True, exist_ok=True)
    tensor = np.empty((1, 3, size, size), dtype=np.float32)
    records, paths = [], []
    for index, path in enumerate(images):
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Cannot decode calibration image: {path}")
        meta = preprocess(image, tensor)
        destination = output / f"input_{index:05d}.npz"
        np.savez(destination, **{input_name: tensor})
        paths.append(str(destination))
        records.append(dict(source=str(path), npz=str(destination), original_hw=meta.original_hw,
                            gain=meta.gain, left=meta.left, top=meta.top))
    listing = output / "data_list.txt"
    listing.write_text("\n".join(paths) + "\n", encoding="utf-8")
    (output / "preprocess.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    return listing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--input-name", default="images")
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    print(prepare(args.dataset, args.output, args.input_name, args.size, args.limit))


if __name__ == "__main__":
    main()
