"""YOLOv8 RGB letterbox preprocessing and decoded-head postprocessing."""

from dataclasses import dataclass

import cv2
import numpy as np

"""YOLOv8 RGB letterbox preprocessing and decoded-head postprocessing."""

@dataclass(frozen=True)
class Letterbox:
    original_hw: tuple #原始高宽
    gain: float #缩放因子
    left: int #左侧padding
    top: int #顶部padding


def preprocess(image, destination):
    """Fill a leased [1,3,H,W] float32 buffer; input images use OpenCV BGR."""
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("Expected a non-empty HWC uint8 BGR image")
    if destination.ndim != 4 or destination.shape[:2] != (1, 3):
        raise ValueError("Expected an NCHW batch-one RGB destination")
    if destination.dtype != np.float32:
        raise ValueError("Preprocessing destination must be float32")
    height, width = image.shape[:2]
    if min(height, width) == 0:
        raise ValueError("Cannot preprocess an empty image")
    target_h, target_w = destination.shape[2:]
    gain = min(target_w / width, target_h / height)
    new_w, new_h = round(width * gain), round(height * gain)
    left = round((target_w - new_w) / 2 - 0.1)
    top = round((target_h - new_h) / 2 - 0.1)
    rgb = cv2.cvtColor(cv2.resize(image, (new_w, new_h)), cv2.COLOR_BGR2RGB)
    destination.fill(114 / 255)
    destination[0, :, top:top + new_h, left:left + new_w] = rgb.transpose(2, 0, 1) / 255
    return Letterbox((height, width), gain, left, top)


def box_iou(box, others):
    intersection_hw = np.maximum(0, np.minimum(box[2:], others[:, 2:]) -
                                  np.maximum(box[:2], others[:, :2]))
    intersection = intersection_hw.prod(axis=1)
    area = np.maximum(0, box[2:] - box[:2]).prod()
    areas = np.maximum(0, others[:, 2:] - others[:, :2]).prod(axis=1)
    return intersection / np.maximum(area + areas - intersection, 1e-12)


def postprocess(output, meta, classes=80, confidence=0.25, iou=0.45, max_detections=300,
                max_candidates=30000):
    """Return [x1,y1,x2,y2,score,class] rows using class-aware, best-class NMS.

    This contract is the Ultralytics decoded detection head, without objectness,
    logits, embedded NMS, or raw DFL branches. Reject other heads explicitly.
    """
    if classes <= 0 or not 0 <= confidence <= 1 or not 0 <= iou <= 1:
        raise ValueError("Invalid class count or NMS thresholds")
    if max_candidates <= 0 or max_detections <= 0:
        raise ValueError("Candidate and detection limits must be positive")
    data = np.asarray(output, dtype=np.float32)
    if data.ndim != 3 or data.shape[0] != 1 or data.shape[1] != classes + 4:
        raise ValueError(f"Expected [1,{classes + 4},N] YOLOv8 output, got {data.shape}")
    rows = data[0].T
    if not np.isfinite(rows).all():
        raise ValueError("Model output contains NaN or infinity")
    probabilities = rows[:, 4:]
    if probabilities.size and (probabilities.min() < -1e-5 or probabilities.max() > 1 + 1e-5):
        raise ValueError("Expected decoded class probabilities in [0,1], not logits")
    labels = probabilities.argmax(axis=1)
    scores = probabilities[np.arange(len(rows)), labels]
    valid = (scores >= confidence) & (rows[:, 2] > 0) & (rows[:, 3] > 0)
    rows, scores, labels = rows[valid], scores[valid], labels[valid]
    if len(rows) == 0:
        return np.empty((0, 6), dtype=np.float32)
    order = np.argsort(-scores, kind="stable")[:max_candidates]
    rows, scores, labels = rows[order], scores[order], labels[order]
    boxes = np.column_stack((rows[:, :2] - rows[:, 2:4] / 2,
                             rows[:, :2] + rows[:, 2:4] / 2))
    remaining = np.arange(len(boxes))
    keep = []
    while remaining.size and len(keep) < max_detections:
        current = remaining[0]
        keep.append(current)
        rest = remaining[1:]
        suppressed = ((labels[rest] == labels[current]) &
                      (box_iou(boxes[current], boxes[rest]) > iou))
        remaining = rest[~suppressed]
    boxes = boxes[keep]
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - meta.left) / meta.gain
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - meta.top) / meta.gain
    original_h, original_w = meta.original_hw
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, original_w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, original_h)
    return np.column_stack((boxes, scores[keep], labels[keep])).astype(np.float32)
