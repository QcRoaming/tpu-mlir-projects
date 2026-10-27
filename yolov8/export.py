"""Export existing local Ultralytics detection weights, without auto-install/download."""

import argparse
import importlib.util
import os
from pathlib import Path
import shutil


def export(weights, output, size=640):
    weights, output = Path(weights).resolve(), Path(output).resolve()
    if not weights.is_file() or weights.suffix != ".pt":
        raise ValueError("A local .pt file is required; model names/URLs are not accepted")
    for dependency in ("ultralytics", "onnx", "torch"):
        if importlib.util.find_spec(dependency) is None:
            raise RuntimeError(f"Missing locally installed {dependency}; no auto-install is used")
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ["HF_HUB_OFFLINE"] = "1"
    output.parent.mkdir(parents=True, exist_ok=True)
    # Ultralytics places exports next to the weights, so isolate a local copy.
    copied = output.parent / "local_export_weights.pt"
    if weights != copied:
        shutil.copy2(weights, copied)
    from ultralytics import YOLO
    model = YOLO(str(copied), task="detect")
    if model.task != "detect":
        raise ValueError("Require a YOLO detection model")
    generated = Path(model.export(format="onnx", imgsz=size, batch=1, dynamic=False,
                                  simplify=False, nms=False, opset=13, device="cpu", half=False))
    if generated.resolve() != output:
        shutil.copy2(generated, output)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=640)
    args = parser.parse_args()
    print(export(args.weights, args.output, args.size))
