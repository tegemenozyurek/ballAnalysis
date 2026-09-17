from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from live_football_vision.model import (
    BALL_CLASS_ID,
    CLASS_LABELS,
    DETECT_CLASS_IDS,
    PLAYER_CLASS_ID,
)

_CONFIDENCE_THRESHOLD = 0.2
_IMAGE_SIZE = 640
_MAX_DETECTIONS = 40


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass(frozen=True)
class DetectionResult:
    detections: tuple[Detection, ...]
    latency_ms: float
    device: str


class FootballDetector:
    """Runs player/ball detection on a BGR frame. No tracking."""

    def __init__(self, model, device: str) -> None:
        self._model = model
        self._device = device

    @property
    def device(self) -> str:
        return self._device

    def warmup(self, width: int = 640, height: int = 384) -> None:
        dummy = np.zeros((height, width, 3), dtype=np.uint8)
        self.detect(dummy)

    def detect(self, frame: np.ndarray) -> DetectionResult:
        start = time.perf_counter()
        results = self._model.predict(
            source=frame,
            device=self._device,
            classes=list(DETECT_CLASS_IDS),
            conf=_CONFIDENCE_THRESHOLD,
            imgsz=_IMAGE_SIZE,
            max_det=_MAX_DETECTIONS,
            verbose=False,
        )
        latency_ms = (time.perf_counter() - start) * 1000.0
        height, width = frame.shape[:2]
        detections = _detections_from_result(results[0], width, height)
        return DetectionResult(
            detections=detections,
            latency_ms=latency_ms,
            device=self._device,
        )


def _detections_from_result(result, frame_width: int, frame_height: int) -> tuple[Detection, ...]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return ()

    xyxy = boxes.xyxy.detach().cpu().numpy()
    class_ids = boxes.cls.detach().cpu().numpy().astype(int)
    confidences = boxes.conf.detach().cpu().numpy()
    return parse_detections(xyxy, class_ids, confidences, frame_width, frame_height)


def parse_detections(
    xyxy: np.ndarray,
    class_ids: np.ndarray,
    confidences: np.ndarray,
    frame_width: int,
    frame_height: int,
) -> tuple[Detection, ...]:
    detections: list[Detection] = []
    for box, class_id, confidence in zip(xyxy, class_ids, confidences, strict=False):
        label = CLASS_LABELS.get(int(class_id))
        if label is None:
            continue
        x1, y1, x2, y2 = (int(round(value)) for value in box[:4])
        x1 = _clip(x1, 0, frame_width - 1)
        x2 = _clip(x2, 0, frame_width - 1)
        y1 = _clip(y1, 0, frame_height - 1)
        y2 = _clip(y2, 0, frame_height - 1)
        if x2 <= x1 or y2 <= y1:
            continue
        detections.append(
            Detection(
                label=label,
                confidence=float(confidence),
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
            )
        )
    return tuple(detections)


def count_labels(detections: tuple[Detection, ...]) -> tuple[int, int]:
    players = sum(1 for item in detections if item.label == CLASS_LABELS[PLAYER_CLASS_ID])
    balls = sum(1 for item in detections if item.label == CLASS_LABELS[BALL_CLASS_ID])
    return players, balls


def _clip(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))
