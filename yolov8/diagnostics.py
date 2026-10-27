"""Compare matching dumped tensors and suggest sensitive float-layer candidates."""

import argparse
import json
from pathlib import Path

import numpy as np


def compare(reference, candidate, threshold=0.99):
    with np.load(reference, allow_pickle=False) as baseline, \
            np.load(candidate, allow_pickle=False) as quant:
        missing = sorted(set(baseline.files) - set(quant.files))
        unexpected = sorted(set(quant.files) - set(baseline.files))
        results = []
        for name in sorted(set(baseline.files) & set(quant.files)):
            left, right = baseline[name], quant[name]
            if left.shape != right.shape:
                results.append(dict(name=name, shape_match=False, reference_shape=left.shape,
                                    candidate_shape=right.shape, sensitive=True))
                continue
            a, b = left.astype(np.float64).ravel(), right.astype(np.float64).ravel()
            if not np.isfinite(a).all() or not np.isfinite(b).all():
                results.append(dict(name=name, shape_match=True, finite=False, sensitive=True))
                continue
            norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
            if norm_a == 0 or norm_b == 0:
                cosine = 1.0 if np.array_equal(a, b) else 0.0
            else:
                cosine = float(np.dot(a / norm_a, b / norm_b).clip(-1, 1))
            error = a - b
            rmse = float(np.sqrt(np.mean(error * error))) if a.size else 0.0
            results.append(dict(name=name, shape_match=True, finite=True, cosine=cosine,
                                rmse=rmse, relative_l2=float(np.linalg.norm(error) /
                                                           max(norm_a, 1e-12)),
                                max_abs=float(np.max(np.abs(error))) if a.size else 0.0,
                                sensitive=cosine < threshold))
    return dict(threshold=threshold, missing=missing, unexpected=unexpected, tensors=results,
                note="Tensor similarity is not COCO mAP; "
                     "proposed names must be MLIR operation names")


def write_qtable(report, path, mode="F16"):
    if mode not in {"F16", "F32", "BF16"}:
        raise ValueError("Unsupported float mode")
    names = [row["name"] for row in report["tensors"] if row["sensitive"] and
             row.get("shape_match") and row.get("finite")]
    if any(any(char.isspace() for char in name) or name.startswith("#") for name in names):
        raise ValueError("Tensor names are not directly representable as qtable operation names")
    Path(path).write_text("# Review tensor names against Top MLIR operation names before deploy\n" +
                          "".join(f"{name} {mode}\n" for name in names), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.99)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--qtable", type=Path)
    args = parser.parse_args()
    report = compare(args.reference, args.candidate, args.threshold)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.qtable:
        write_qtable(report, args.qtable)
