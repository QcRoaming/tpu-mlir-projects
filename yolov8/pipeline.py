"""Four bounded stages with explicit frame ownership and cooperative cancellation."""

from collections import deque
from dataclasses import dataclass
import threading
import time

import cv2
import numpy as np

from .vision import preprocess, postprocess


class EndOfChannel(Exception):
    pass

# 管理有界队列
class Channel:
    """A closed channel drains normally; an aborted channel discards immediately."""

    def __init__(self, capacity, policy, dispose):
        if capacity < 1 or policy not in {"block", "drop-oldest"}:
            raise ValueError("Require positive queue capacity and block/drop-oldest policy")
        self.capacity, self.policy, self.dispose = capacity, policy, dispose
        self.items = deque()
        self.condition = threading.Condition()
        self.closed = False
    # 队列满时阻塞
    def put(self, item, stop):
        evicted = None
        with self.condition:
            while len(self.items) >= self.capacity and not self.closed and not stop.is_set():
                if self.policy == "drop-oldest": #队列满时的丢帧策略，先移除最旧的packet（旧帧）再放入新的packet
                    evicted = self.items.popleft()
                    break
                self.condition.wait(0.05)
            if self.closed or stop.is_set():
                accepted = False
            else:
                self.items.append(item)
                self.condition.notify_all()
                accepted = True
        if evicted is not None:
            self.dispose(evicted)
        if not accepted:
            self.dispose(item)
        return accepted

    def get(self, stop):
        with self.condition:
            while not self.items and not self.closed and not stop.is_set():
                self.condition.wait(0.05)
            if stop.is_set() or not self.items:
                raise EndOfChannel()
            item = self.items.popleft()
            self.condition.notify_all()
            return item

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()

    def abort(self):
        with self.condition:
            self.closed = True
            items = list(self.items)
            self.items.clear()
            self.condition.notify_all()
        for item in items:
            self.dispose(item)

# 缓冲区池：
class BufferPool:
    """A lease is held until synchronous inference finishes or the packet drops."""

    def __init__(self, count, shape):
        if count < 1:
            raise ValueError("Buffer count must be positive")
        self.free = deque(np.empty(shape, dtype=np.float32) for _ in range(count))
        self.busy = set()
        self.condition = threading.Condition()

    def acquire(self, stop):
        with self.condition:
            while not self.free and not stop.is_set():
                self.condition.wait(0.05)
            if stop.is_set():
                raise EndOfChannel()
            array = self.free.popleft()
            self.busy.add(id(array))
            return array

    def release(self, array):
        with self.condition:
            if id(array) not in self.busy:
                raise RuntimeError("Tensor buffer was released twice or without a lease")
            self.busy.remove(id(array))
            self.free.append(array)
            self.condition.notify_all()
#从视频中读出一帧
class VideoSource:
    """Open one local video or explicit RTSP URL in its own decode worker."""

    def __init__(self, source, max_frames=0):
        self.source, self.max_frames = source, max_frames
        self.capture = None

    def __iter__(self):
        self.capture = cv2.VideoCapture(self.source)
        if not self.capture.isOpened():
            raise RuntimeError(f"Cannot open video source: {self.source}")
        count = 0
        while self.max_frames == 0 or count < self.max_frames:
            ok, frame = self.capture.read()
            if not ok:
                break
            count += 1
            yield frame

    def close(self):
        if self.capture is not None:
            self.capture.release()


class SyntheticSource:
    """Finite local frames for a smoke test; no real detections are implied."""

    def __init__(self, count=16, size=(240, 320), stream=0):
        self.count, self.size, self.stream = count, size, stream

    def __iter__(self):
        height, width = self.size
        for index in range(self.count):
            image = np.zeros((height, width, 3), dtype=np.uint8)
            image[..., 0] = (index * 11 + self.stream * 23) % 256
            image[height // 4:height * 3 // 4, width // 4:width * 3 // 4, 1] = 220
            yield image

    def close(self):
        pass

# 用于存放一帧画面的数据
@dataclass
class Packet:
    stream: str
    index: int
    image: np.ndarray
    captured: float
    tensor: object = None
    meta: object = None
    output: object = None
    stage_ms: object = None


class Pipeline:
    """Decode producers -> preprocess -> inference -> postprocess, all bounded.

    Stats count every decoded frame as submitted, then completed or dropped.
    Latency starts after decoding; video decode/network delay is not included.
    Native capture/inference calls cannot be cancelled safely by Python threads.
    """

    def __init__(self, backend, sources, queue_size=4, policy="block", buffers=None,
                 classes=80, confidence=0.25, iou=0.45, sink=None):
        if not sources:
            raise ValueError("At least one source is required")
        self.backend, self.sources, self.sink = backend, sources, sink
        self.classes, self.confidence, self.iou = classes, confidence, iou
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.pool = BufferPool(queue_size + 2 if buffers is None else buffers, backend.input_shape)
        self.stats = {name: dict(submitted=0, completed=0, dropped=0, stage_ms={},
                                latency_ms=deque(maxlen=4096)) for name in sources}
        # 初始化创建三个有界队列，分别对应三个阶段
        # _decode()
        #     ↓ queues[0]
        # _preprocess()
        #     ↓ queues[1]
        # _infer()
        #     ↓ queues[2]
        # _postprocess()
        self.queues = [Channel(queue_size, policy, self._dispose) for _ in range(3)]
        self.producers_remaining = len(sources)
        self.errors = []
        self.threads = []
        self.started = None
        self.finished = None
        self._has_run = False

    def _dispose(self, packet):
        if packet.tensor is not None:
            self.pool.release(packet.tensor)
            packet.tensor = None
        with self.lock:
            self.stats[packet.stream]["dropped"] += 1

    def cancel(self):
        self.stop.set()
        for channel in self.queues:
            channel.abort()

    def _failed(self, stage, error):
        with self.lock:
            self.errors.append((stage, error))
        self.cancel()
#将图像放进packet，然后放进队列中，等待后续处理 ，生产者
    def _decode(self, name, source):
        try:
            for index, image in enumerate(source):
                if self.stop.is_set():
                    break
                packet = Packet(name, index, image, time.perf_counter(), stage_ms={})
                with self.lock:
                    self.stats[name]["submitted"] += 1
                if not self.queues[0].put(packet, self.stop):
                    break
        except Exception as error:
            self._failed(f"decode/{name}", error)
        finally:
            try:
                source.close()
            except Exception as error:
                self._failed(f"close/{name}", error)
            with self.lock:
                self.producers_remaining -= 1
                last = self.producers_remaining == 0
            if last:
                self.queues[0].close()

    def _stage(self, index, operation):
        channel = self.queues[index]
        try:
            while True:
                packet = channel.get(self.stop)
                try:
                    operation(packet)
                except BaseException:
                    self._dispose(packet)
                    raise
                if index < 2:
                    if not self.queues[index + 1].put(packet, self.stop):
                        break
        except EndOfChannel:
            pass
        except Exception as error:
            self._failed(("preprocess", "inference", "postprocess")[index], error)
        finally:
            if index < 2:
                self.queues[index + 1].close()
#预处理线程取出packet，并进行前处理
    def _preprocess(self, packet):
        packet.tensor = self.pool.acquire(self.stop)
        start = time.perf_counter()
        packet.meta = preprocess(packet.image, packet.tensor)
        packet.stage_ms["preprocess"] = (time.perf_counter() - start) * 1000
#推理过程
    def _infer(self, packet):
        start = time.perf_counter()
        try:
            # Copy before releasing the input lease; some runtimes return views.
            #self.backend里面是模型文件+对应运行时的推理封装
            packet.output = np.array(self.backend.infer(packet.tensor), copy=True)
        finally:
            self.pool.release(packet.tensor)
            packet.tensor = None
        packet.stage_ms["inference"] = (time.perf_counter() - start) * 1000

    def _postprocess(self, packet):
        start = time.perf_counter()
        detections = postprocess(packet.output, packet.meta, self.classes,
                                 self.confidence, self.iou)
        packet.stage_ms["postprocess"] = (time.perf_counter() - start) * 1000
        if self.sink is not None:
            self.sink(packet, detections)
        latency = (time.perf_counter() - packet.captured) * 1000
        with self.lock:
            stats = self.stats[packet.stream]
            stats["completed"] += 1
            stats["latency_ms"].append(latency)
            for stage, value in packet.stage_ms.items():
                stats["stage_ms"][stage] = stats["stage_ms"].get(stage, 0) + value

    def run(self, timeout=None):
        if self._has_run:
            raise RuntimeError("Pipeline instances can run only once")
        self._has_run = True
        self.started = time.perf_counter()
        # 创建三个阶段的线程
        # yolo-stage-0 -> _preprocess()
        # yolo-stage-1 -> _infer()
        # yolo-stage-2 -> _postprocess()
        for index, operation in enumerate((self._preprocess, self._infer, self._postprocess)):
            self.threads.append(threading.Thread(target=self._stage, args=(index, operation),
                                                 name=f"yolo-stage-{index}", daemon=True))
            # 给每个视频源创建解码线程
        for name, source in self.sources.items():
            self.threads.append(threading.Thread(target=self._decode, args=(name, source),
                                                 name=f"yolo-decode-{name}", daemon=True))
        for thread in self.threads:
            thread.start() #统一启动
        deadline = None if timeout is None else self.started + timeout
        try:
            for thread in self.threads:
                while thread.is_alive():
                    thread.join(0.05)
                    if deadline is not None and time.perf_counter() >= deadline:
                        self.cancel()
                        raise TimeoutError("Pipeline timeout: a native capture/inference call "
                                           "may still be running; use a separate process for "
                                           "hard cancellation")
        except KeyboardInterrupt:
            self.cancel()
            for thread in self.threads:
                thread.join(1)
            raise
        self.finished = time.perf_counter()
        if self.pool.busy:
            raise RuntimeError("Tensor leases leaked at shutdown")
        if self.errors:
            stage, error = self.errors[0]
            raise RuntimeError(f"Pipeline failed in {stage}: {error}") from error
        return self.report()

    def report(self):
        end = self.finished or time.perf_counter()
        elapsed = max(end - (self.started or end), 1e-9)
        with self.lock:
            streams = {}
            for name, stats in self.stats.items():
                count = stats["completed"]
                samples = list(stats["latency_ms"])
                streams[name] = {
                    key: stats[key] for key in ("submitted", "completed", "dropped")
                }
                streams[name].update(
                    fps=count / elapsed,
                    latency_p50_ms=float(np.percentile(samples, 50)) if samples else None,
                    latency_p95_ms=float(np.percentile(samples, 95)) if samples else None,
                    latency_sample_count=len(samples),
                    mean_stage_ms={stage: total / count for stage, total in
                                   stats["stage_ms"].items()} if count else {})
            return dict(elapsed_seconds=elapsed,
                        aggregate_fps=sum(s["completed"] for s in streams.values()) / elapsed,
                        latency_definition="after decode through sink completion, "
                                           "last 4096 samples",
                        streams=streams)
