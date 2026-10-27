"""Generate a tiny input-dependent YOLO-shaped ONNX fixture, not YOLOv8 weights."""

import argparse
from pathlib import Path

import numpy as np


def create(path, size=64, classes=80, candidates=16):
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    if size < 32 or classes < 1 or candidates < 2:
        raise ValueError("Invalid fixture dimensions")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # A compact pooled convolution is input-dependent and compiler-compatible.
    weights = np.zeros((classes + 4, 3, 1, 1), dtype=np.float32)
    weights[4, 1, 0, 0] = 0.1
    bias = np.zeros(classes + 4, dtype=np.float32)
    bias[:4] = [size / 2, size / 2, size / 2, size / 2]
    bias[4] = 0.7
    shape = np.array([1, classes + 4, 1], dtype=np.int64)
    copies = np.array([1, 1, candidates], dtype=np.int64)
    nodes = [helper.make_node("GlobalAveragePool", ["images"], ["pool"]),
             helper.make_node("Conv", ["pool", "weights", "bias"], ["head"],
                              kernel_shape=[1, 1]),
             helper.make_node("Reshape", ["head", "shape"], ["reshaped"]),
             helper.make_node("Tile", ["reshaped", "copies"], ["output0"])]
    graph = helper.make_graph(nodes, "synthetic_yolo_contract",
                              [helper.make_tensor_value_info("images", TensorProto.FLOAT,
                                                             [1, 3, size, size])],
                              [helper.make_tensor_value_info("output0", TensorProto.FLOAT,
                                                             [1, classes + 4, candidates])],
                              [numpy_helper.from_array(array, name) for name, array in
                               (("weights", weights), ("bias", bias), ("shape", shape),
                                ("copies", copies))])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)],
                              producer_name="local-synthetic-fixture", ir_version=8)
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--classes", type=int, default=80)
    parser.add_argument("--candidates", type=int, default=16)
    args = parser.parse_args()
    print(create(args.output, args.size, args.classes, args.candidates))
