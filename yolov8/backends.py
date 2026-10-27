"""Lazy CPU, CModel and device backends with a batch-one decoded-head contract."""

import ctypes
import importlib
import os
from pathlib import Path
import subprocess
import threading

import numpy as np


class OnnxBackend:

    def __init__(self, model, output_name=None):
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        self.session = ort.InferenceSession(str(model), options,
                                            providers=["CPUExecutionProvider"])
        inputs = self.session.get_inputs()
        if len(inputs) != 1 or inputs[0].type != "tensor(float)":
            raise ValueError("Require one float32 ONNX input")
        self.input_name = inputs[0].name
        self.input_shape = tuple(inputs[0].shape)
        if (len(self.input_shape) != 4 or self.input_shape[:2] != (1, 3) or
                not all(isinstance(x, int) and x > 0 for x in self.input_shape)):
            raise ValueError("Require static [1,3,H,W] input; export dynamic=False, batch=1")
        outputs = self.session.get_outputs()
        if output_name is None and len(outputs) != 1:
            raise ValueError("Select --output-name when the model has multiple outputs")
        self.output_name = output_name or outputs[0].name
        if self.output_name not in {x.name for x in outputs}:
            raise ValueError("Unknown ONNX output name")

    def infer(self, tensor):
        return self.session.run([self.output_name], {self.input_name: tensor})[0]


_NATIVE_MODE = None
_NATIVE_LOCK = threading.Lock()
_NATIVE_HANDLES = []


class BmodelBackend:
    """Keep one pyruntime_bm Net; call it from the single inference worker only.

    Native runtime libraries are process-global. Existing installed aliases must
    match the requested mode; this sample never rewrites installed libraries.
    """

    def __init__(self, model, mode="cmodel", device=0, output_name=None, chip="bm1684x"):
        if chip != "bm1684x" or mode not in {"cmodel", "device"}:
            raise ValueError("This SC7 HP75 example supports bm1684x cmodel/device only")
        model = Path(model).resolve()
        if not model.is_file() or model.suffix != ".bmodel":
            raise ValueError("Expected a local .bmodel file")
        root = os.environ.get("TPUC_ROOT")
        if not root:
            raise RuntimeError("Set TPUC_ROOT and LD_LIBRARY_PATH to the local TPU runtime")
        libraries = Path(root) / "lib"
        alias_dir = Path(os.environ.get("YOLO_CMODEL_LIBRARY_DIR", libraries))
        alias = alias_dir / "libcmodel.so"
        selected = libraries / "libcmodel_bm1684x.so"
        if mode == "cmodel":
            if not selected.is_file() or alias.resolve() != selected.resolve():
                raise RuntimeError("libcmodel.so must select libcmodel_bm1684x.so; run this "
                                   "sample's CLI to prepare isolated CModel aliases")
        else:
            if device < 0 or not list(Path("/dev").glob("bm-sophon*")):
                raise RuntimeError("No Sophon device node found; use --backend cmodel or onnx")
            if os.environ.get("USING_CMODEL", "True").lower() not in {"false", "0"}:
                raise RuntimeError("Set USING_CMODEL=False and select the real Sophon SDK "
                                   "library directory for --backend device")
            # Resolve the library actually chosen by LD_LIBRARY_PATH. A specific
            # simulator export is evidence; a log string containing cmodel is not.
            loaded = ctypes.CDLL("libbmlib.so.0", mode=ctypes.RTLD_GLOBAL)
            _NATIVE_HANDLES.append(loaded)
            try:
                ctypes.c_void_p.in_dll(loaded, "cmodel_so_handle_")
            except ValueError:
                pass
            else:
                raise RuntimeError("The loaded bmlib exports the CModel-only cmodel_so_handle_; "
                                   "use the actual Sophon device SDK/runtime")
        metadata = subprocess.run(["model_tool", "--chip", str(model)], check=True,
                                  capture_output=True, text=True)
        actual = metadata.stdout.strip().lower()
        if actual != chip:
            raise ValueError(f"BModel chip {actual} does not match requested {chip}")
        global _NATIVE_MODE
        with _NATIVE_LOCK:
            if _NATIVE_MODE is not None and _NATIVE_MODE != (mode, chip):
                raise RuntimeError("CModel/device/chip switching requires a new Python process")
            if mode == "cmodel":
                _NATIVE_HANDLES.append(ctypes.CDLL(str(selected), mode=ctypes.RTLD_GLOBAL))
            runtime = importlib.import_module("pyruntime_bm")
            self.model = runtime.Model(str(model), device, "")
            _NATIVE_MODE = (mode, chip)
        self.net = self.model.Net(self.model.networks[0])
        if len(self.net.inputs) != 1:
            raise ValueError("Require one BModel input")
        self.input = self.net.inputs[0]
        self.input_shape = tuple(self.input.data.shape)
        if len(self.input_shape) != 4 or self.input_shape[:2] != (1, 3):
            raise ValueError("Require batch-one RGB NCHW BModel input, without fused preprocessing")
        outputs = self.net.outputs
        if output_name is None and len(outputs) != 1:
            raise ValueError("Select --output-name when the model has multiple outputs")
        name = output_name or outputs[0].name
        self.output = next((x for x in outputs if x.name == name), None)
        if self.output is None:
            raise ValueError("Unknown BModel output name")
        self._infer_lock = threading.Lock()

    def infer(self, tensor):
        from utils.lowering import bf16_to_fp32, lowering
        with self._infer_lock:
            self.input.data[...] = lowering(tensor, pdtype=self.input.dtype,
                                           pshape=self.input_shape,
                                           pzero_point=self.input.qzero_point,
                                           pscale=self.input.qscale)
            self.net.forward()
            output = self.output.data
            if self.output.dtype in {"i8", "u8"}:
                if self.output.qscale <= 0:
                    raise ValueError("Invalid quantized output scale")
                return (output.astype(np.float32) - self.output.qzero_point) * self.output.qscale
            if self.output.dtype == "bf16":
                return bf16_to_fp32(output).copy()
            return output.astype(np.float32, copy=True)
