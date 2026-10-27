"""Plan or execute local-only YOLOv8 ONNX -> F16/INT8 BModel workflows."""

import argparse
from pathlib import Path
import sys

if __package__ in {None, "", "yolov8"}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from common import Stage, require_local_dir, require_local_file, run_workflow, tool_command
else:
    from ..common import Stage, require_local_dir, require_local_file, run_workflow, tool_command


def build_stages(args):
    directory = args.output.resolve()
    scripts = Path(__file__).resolve().parent
    stages = []
    if args.weights:
        weights = require_local_file(args.weights)
        onnx = directory / "yolov8.onnx"
        stages.append(Stage("export_local_pt", (sys.executable, str(scripts / "export.py"),
                                                "--weights", str(weights), "--output", str(onnx),
                                                "--size", str(args.size)), directory))
    else:
        onnx = require_local_file(args.onnx)
    if args.dataset:
        dataset = require_local_dir(args.dataset)
        stages.append(Stage("prepare_exact_inputs", (sys.executable, str(scripts / "prepare.py"),
                                                     "--dataset", str(dataset), "--output",
                                                     str(directory / "inputs"), "--input-name",
                                                     args.input_name, "--size", str(args.size),
                                                     "--limit", str(args.input_num)), directory))
    elif args.precision != "F16" or args.search_mixed:
        raise ValueError("INT8 calibration and mixed search require --dataset local images")
    mlir = directory / "yolov8.mlir"
    transform = list(tool_command("model_transform.py", "--model_name", "yolov8",
                                  "--model_def", str(onnx), "--input_shapes",
                                  f"[[1,3,{args.size},{args.size}]]", "--mlir", str(mlir)))
    test_input = directory / "inputs/input_00000.npz"
    reference = directory / "yolov8_top_outputs.npz"
    if args.dataset:
        transform.extend(("--test_input", str(test_input), "--test_result", str(reference)))
    stages.append(Stage("transform_onnx", tuple(transform), directory))
    calibration = directory / f"calibration_{args.calibration}.table"
    qtable = require_local_file(args.qtable) if args.qtable else None
    if args.precision == "INT8" or args.search_mixed:
        command = list(tool_command("run_calibration.py", str(mlir), "--data_list",
                                     str(directory / "inputs/data_list.txt"), "--input_num",
                                     str(args.input_num), "--tune_num", "0", "--cali_method",
                                     args.calibration, "-o", str(calibration)))
        if args.search_mixed:
            qtable = directory / "mixed_precision.qtable"
            command.extend(("--search", "search_qtable", "--processor", "bm1684x",
                            "--fp_type", "F16", "--mix_mode", "wi8ai8_fp", "--inference_num",
                            str(args.input_num), "--quantize_method_list", args.calibration,
                            "--expected_cos", str(args.expected_cos), "--quantize_table",
                            str(qtable)))
            if args.qtable:
                command.extend(("--pre_qtable", str(require_local_file(args.qtable))))
        stages.append(Stage("calibrate_and_search" if args.search_mixed else "calibrate",
                            tuple(command), directory))
    name = f"yolov8_bm1684x_{args.precision.lower()}"
    if qtable:
        name += "_mixed"
    deploy = list(tool_command("model_deploy.py", "--mlir", str(mlir), "--processor",
                               "bm1684x", "--quantize", args.precision, "--model",
                               str(directory / f"{name}.bmodel")))
    if args.precision == "INT8":
        deploy.extend(("--calibration_table", str(calibration)))
    if qtable:
        if args.precision != "INT8":
            raise ValueError("A qtable/search-mixed requires --precision INT8")
        deploy.extend(("--quantize_table", str(qtable)))
    if args.dataset:
        deploy.extend(("--test_input", str(test_input), "--test_reference", str(reference),
                       "--tolerance", "0.85,0.45" if args.precision == "INT8" else "0.99,0.90"))
    if args.skip_validation:
        deploy.append("--skip_validation")
    stages.append(Stage("deploy_bmodel", tuple(deploy), directory))
    return stages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    model = parser.add_mutually_exclusive_group(required=True)
    model.add_argument("--onnx", type=Path)
    model.add_argument("--weights", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--input-name", default="images")
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--input-num", type=int, default=100)
    parser.add_argument("--precision", choices=("F16", "INT8"), default="F16")
    parser.add_argument("--calibration", choices=("max", "kl"), default="kl")
    parser.add_argument("--qtable", type=Path)
    parser.add_argument("--search-mixed", action="store_true")
    parser.add_argument("--expected-cos", type=float, default=0.99)
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--execute", action="store_true",
                        help="Run compiler tools; otherwise only write the reviewable plan")
    args = parser.parse_args()
    if args.size < 32 or args.size % 32 or args.input_num <= 0:
        parser.error("--size must be a positive multiple of 32; --input-num must be positive")
    print(run_workflow(build_stages(args), args.output, args.execute))


if __name__ == "__main__":
    main()
