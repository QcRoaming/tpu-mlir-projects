#!/usr/bin/env python3
"""Exercise the detection pipeline with a tiny generated model and optional CModel.

No pretrained model is downloaded or loaded. Fixture results establish interface
and pipeline correctness only, not YOLOv8 accuracy or hardware performance.
"""

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import Stage, run_workflow
else:
    from .common import Stage, run_workflow


def prepare_images(directory: Path):
    import cv2
    if __package__ in (None, ""):
        from yolov8.pipeline import SyntheticSource
    else:
        from .yolov8.pipeline import SyntheticSource
    directory.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(SyntheticSource(count=8)):
        if not cv2.imwrite(str(directory / f"{index:03d}.png"), frame):
            raise RuntimeError("Cannot write synthetic calibration image")


def validate_detections(output: Path, backends):
    import numpy as np
    baseline = None
    for backend in backends:
        rows = [json.loads(line) for line in
                (output / f"{backend}_detections.jsonl").read_text().splitlines()]
        by_frame = {(row["stream"], row["frame"]): row["detections"] for row in rows}
        if (len(rows) != 32 or len(by_frame) != 32 or
                any(len(boxes) != 1 for boxes in by_frame.values())):
            raise RuntimeError(f"{backend} did not produce one fixture detection per frame")
        if baseline is None:
            baseline = by_frame
        else:
            if baseline.keys() != by_frame.keys():
                raise RuntimeError("CPU/CModel completed different fixture frames")
            for key in baseline:
                np.testing.assert_allclose(by_frame[key], baseline[key], rtol=1e-3, atol=1e-3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "artifacts/smoke")
    parser.add_argument("--cmodel", action="store_true",
                        help="Also compile and validate F16, then run four streams "
                        "in BM1684X CModel")
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    scripts = Path(__file__).resolve().parent / "yolov8"
    model = output / "fixture.onnx"
    stages = [Stage("generate_fixture", (sys.executable, str(scripts / "fixture.py"),
                                         "--output", str(model)), output)]

    def pipeline_stage(name, backend, model_path):
        return Stage(name, (sys.executable, str(scripts / "run.py"), "--model", str(model_path),
                            "--backend", backend, "--streams", "4", "--synthetic-frames", "8",
                            "--queue-size", "2", "--policy", "block", "--metrics",
                            str(output / f"{backend}_metrics.json"), "--detections",
                            str(output / f"{backend}_detections.jsonl")), output)

    stages.append(pipeline_stage("cpu_four_streams", "onnx", model))
    if args.cmodel:
        prepare_images(output / "images")
        compiled = output / "compiled"
        stages.append(Stage("compile_f16", (sys.executable, str(scripts / "compile.py"),
                                             "--onnx", str(model), "--size", "64", "--dataset",
                                             str(output / "images"), "--input-num", "8",
                                             "--precision", "F16", "--output", str(compiled),
                                             "--execute"), output))
        stages.append(pipeline_stage("cmodel_four_streams", "cmodel",
                                     compiled / "yolov8_bm1684x_f16.bmodel"))
    run_workflow(stages, output / "logs", execute=True)
    backends = ("onnx", "cmodel") if args.cmodel else ("onnx",)
    for backend in backends:
        report = json.loads((output / f"{backend}_metrics.json").read_text())
        streams = report["streams"]
        if report["status"] != "completed" or len(streams) != 4 or any(
                row["completed"] != 8 or row["dropped"] != 0 for row in streams.values()):
            raise RuntimeError(f"{backend} did not complete all 32 fixture frames")
        print(f"{backend}: all 4 streams completed 8 frames with no drops")
    validate_detections(output, backends)
    print(f"Generated fixture only. Logs and metrics: {output}")


if __name__ == "__main__":
    main()
