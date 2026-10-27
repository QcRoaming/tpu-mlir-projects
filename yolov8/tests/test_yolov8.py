"""Behavior tests for geometry, queue ownership, failures and local-only workflows."""

import argparse
import json
from pathlib import Path
import sys
import threading
import time

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from yolov8.backends import OnnxBackend
from yolov8.compile import build_stages
from yolov8.coco_predict import predict
from yolov8.diagnostics import compare
from yolov8.fixture import create
from yolov8.pipeline import BufferPool, Channel, Pipeline, SyntheticSource
from yolov8.prepare import prepare
from yolov8.vision import Letterbox, postprocess, preprocess


class FakeBackend:
    input_shape = (1, 3, 32, 32)

    def __init__(self, delay=0, fail=False):
        self.delay, self.fail = delay, fail

    def infer(self, tensor):
        if self.fail:
            raise ValueError("deliberate inference failure")
        before = tensor.copy()
        time.sleep(self.delay)
        assert np.array_equal(tensor, before), "Input lease reused during inference"
        output = np.zeros((1, 6, 2), dtype=np.float32)
        output[0, :4, :] = np.array([16, 16, 12, 12])[:, None]
        output[0, 4, :] = 0.8
        return output


def test_letterbox_rgb_and_coordinates():
    image = np.zeros((50, 100, 3), dtype=np.uint8)
    image[:] = [10, 20, 30]
    tensor = np.empty((1, 3, 64, 64), np.float32)
    meta = preprocess(image, tensor)
    assert (meta.gain, meta.left, meta.top) == (0.64, 0, 16)
    np.testing.assert_allclose(tensor[0, :, 16, 0], np.array([30, 20, 10]) / 255)
    np.testing.assert_allclose(tensor[0, :, 0, 0], np.full(3, 114 / 255))
    output = np.array([[[32], [32], [32], [16], [0.9], [0.1]]], dtype=np.float32)
    detections = postprocess(output, meta, classes=2)
    np.testing.assert_allclose(detections[0, :4], [25, 12.5, 75, 37.5])


def test_class_aware_nms_and_no_objectness():
    output = np.zeros((1, 6, 3), np.float32)
    output[0, :4, :] = np.array([16, 16, 12, 12])[:, None]
    output[0, 4, :] = [0.9, 0.8, 0.1]
    output[0, 5, :] = [0.1, 0.1, 0.85]
    detections = postprocess(output, Letterbox((32, 32), 1, 0, 0), classes=2)
    assert len(detections) == 2
    assert detections[:, 5].tolist() == [0, 1]


@pytest.mark.parametrize("shape", [(1, 85, 8400), (1, 8400, 84), (2, 84, 8400)])
def test_reject_other_head_shapes(shape):
    with pytest.raises(ValueError, match="Expected"):
        postprocess(np.zeros(shape), Letterbox((64, 64), 1, 0, 0))


def test_buffer_pool_and_drop_queue_ownership():
    stop = threading.Event()
    pool = BufferPool(2, (1, 3, 32, 32))
    first, second = pool.acquire(stop), pool.acquire(stop)
    disposed = []

    def dispose(array):
        disposed.append(id(array))
        pool.release(array)

    channel = Channel(1, "drop-oldest", dispose)
    assert channel.put(first, stop)
    assert channel.put(second, stop)
    assert disposed == [id(first)]
    assert channel.get(stop) is second
    pool.release(second)
    with pytest.raises(RuntimeError, match="twice"):
        pool.release(second)
    assert not pool.busy


def test_block_channel_cancel_releases_waiting_producer():
    stop, disposed = threading.Event(), []
    channel = Channel(1, "block", disposed.append)
    assert channel.put(1, stop)
    result = []
    thread = threading.Thread(target=lambda: result.append(channel.put(2, stop)))
    thread.start()
    time.sleep(0.01)
    stop.set()
    channel.abort()
    thread.join(1)
    assert not thread.is_alive()
    assert result == [False]
    assert sorted(disposed) == [1, 2]


def test_four_stream_pipeline_completes_and_returns_leases():
    sources = {f"cam{i}": SyntheticSource(12, (32, 48), i) for i in range(4)}
    completed = []
    pipeline = Pipeline(FakeBackend(0.001), sources, queue_size=2, buffers=3, classes=2,
                        sink=lambda packet, result: completed.append((packet.stream, packet.index)))
    report = pipeline.run(timeout=5)
    assert len(completed) == 48
    assert report["aggregate_fps"] > 0
    for stats in report["streams"].values():
        assert (stats["submitted"], stats["completed"], stats["dropped"]) == (12, 12, 0)
        assert stats["latency_p95_ms"] >= stats["latency_p50_ms"]
    assert not pipeline.pool.busy


def test_drop_oldest_pipeline_accounts_for_every_submitted_frame():
    sources = {f"cam{i}": SyntheticSource(100, (32, 32), i) for i in range(4)}
    pipeline = Pipeline(FakeBackend(0.003), sources, queue_size=1, buffers=2,
                        policy="drop-oldest", classes=2)
    report = pipeline.run(timeout=5)
    assert sum(stats["completed"] for stats in report["streams"].values()) > 0
    assert sum(stats["dropped"] for stats in report["streams"].values()) > 0
    for stats in report["streams"].values():
        assert stats["submitted"] == stats["completed"] + stats["dropped"] == 100
    assert not pipeline.pool.busy


def test_inference_failure_cancels_workers_without_leaks():
    pipeline = Pipeline(FakeBackend(fail=True), {"cam": SyntheticSource(100)},
                        queue_size=1, buffers=2, classes=2)
    with pytest.raises(RuntimeError, match="deliberate inference failure"):
        pipeline.run(timeout=3)
    assert not any(thread.is_alive() for thread in pipeline.threads)
    assert not pipeline.pool.busy
    stats = pipeline.report()["streams"]["cam"]
    assert stats["submitted"] == stats["completed"] + stats["dropped"]


def test_preprocess_failure_returns_lease():
    class BadSource:
        def __iter__(self):
            yield np.zeros((20, 20), np.uint8)

        def close(self):
            pass

    pipeline = Pipeline(FakeBackend(), {"bad": BadSource()}, queue_size=1, classes=2)
    with pytest.raises(RuntimeError, match="preprocess"):
        pipeline.run(timeout=3)
    assert not pipeline.pool.busy
    assert pipeline.report()["streams"]["bad"]["dropped"] == 1


def test_zero_buffers_rejected():
    with pytest.raises(ValueError, match="positive"):
        Pipeline(FakeBackend(), {"cam": SyntheticSource(1)}, buffers=0)


def test_calibration_inputs_equal_runtime_preprocessing(tmp_path):
    dataset = tmp_path / "images"
    dataset.mkdir()
    image = np.full((20, 40, 3), [17, 33, 99], np.uint8)
    cv2.imwrite(str(dataset / "sample.png"), image)
    listing = prepare(dataset, tmp_path / "inputs", size=64, limit=1)
    expected = np.empty((1, 3, 64, 64), np.float32)
    preprocess(image, expected)
    with np.load(listing.read_text().strip()) as saved:
        np.testing.assert_array_equal(saved["images"], expected)


def test_quantization_diagnostics_zero_vector_and_missing(tmp_path):
    np.savez(tmp_path / "a.npz", zero=np.zeros(3), missing=np.ones(1), equal=np.ones(3))
    np.savez(tmp_path / "b.npz", zero=np.ones(3), equal=np.ones(3))
    report = compare(tmp_path / "a.npz", tmp_path / "b.npz")
    assert report["missing"] == ["missing"]
    assert report["tensors"][0]["cosine"] == pytest.approx(1)
    assert report["tensors"][1]["cosine"] == 0
    assert report["tensors"][1]["sensitive"]


def test_compile_plan_has_local_calibration_and_mixed_qtable(tmp_path, monkeypatch):
    tools = tmp_path / "compiler/python/tools"
    tools.mkdir(parents=True)
    for name in ("model_transform.py", "run_calibration.py", "model_deploy.py"):
        (tools / name).touch()
    monkeypatch.setenv("TPU_MLIR_ROOT", str(tmp_path / "compiler"))
    model = tmp_path / "local onnx.onnx"
    model.write_bytes(b"placeholder")
    images = tmp_path / "images"
    images.mkdir()
    args = argparse.Namespace(output=tmp_path / "compile", weights=None, onnx=model,
                              dataset=images, precision="INT8", calibration="max",
                              input_name="images", input_num=2, size=64, search_mixed=True,
                              qtable=None, expected_cos=0.99, skip_validation=True)
    stages = build_stages(args)
    assert [stage.name for stage in stages] == ["prepare_exact_inputs", "transform_onnx",
                                               "calibrate_and_search", "deploy_bmodel"]
    calibration = stages[2].argv
    assert calibration[calibration.index("--cali_method") + 1] == "max"
    deploy = stages[-1].argv
    assert deploy[deploy.index("--processor") + 1] == "bm1684x"
    assert "--quantize_table" in deploy
    assert all("http" not in argument for stage in stages for argument in stage.argv)


def test_input_dependent_synthetic_onnx_runs_full_pipeline(tmp_path):
    pytest.importorskip("onnxruntime")
    model = create(tmp_path / "fixture.onnx", size=64)
    backend = OnnxBackend(model)
    output_a = backend.infer(np.zeros(backend.input_shape, np.float32))
    output_b = backend.infer(np.ones(backend.input_shape, np.float32))
    assert output_a.shape == (1, 84, 16)
    assert not np.array_equal(output_a, output_b)
    pipeline = Pipeline(backend, {f"cam{i}": SyntheticSource(3, stream=i) for i in range(4)})
    report = pipeline.run(timeout=5)
    assert sum(stats["completed"] for stats in report["streams"].values()) == 12


def test_coco_prediction_category_mapping_and_image_subset(tmp_path):
    image = np.zeros((20, 40, 3), np.uint8)
    cv2.imwrite(str(tmp_path / "sample.png"), image)
    annotations = tmp_path / "annotations.json"
    annotations.write_text(json.dumps(dict(images=[dict(id=17, file_name="sample.png")],
                                            categories=[dict(id=1), dict(id=3)])))
    predictions, ids = predict(FakeBackend(), tmp_path, annotations, category_ids=(3, 1))
    assert ids == [17]
    assert len(predictions) == 1
    assert predictions[0]["category_id"] == 3
    assert predictions[0]["image_id"] == 17
    np.testing.assert_allclose(predictions[0]["bbox"], [12.5, 2.5, 15, 15])
