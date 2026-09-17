from __future__ import annotations

from collections import deque
from collections.abc import Callable
import time

import cv2
import numpy as np

from live_football_vision.detect import Detection, DetectionResult, FootballDetector, count_labels
from live_football_vision.region import Region

_WINDOW_NAME = "Live Football Vision — Onizleme"
_FPS_WINDOW = 30
_PLAYER_COLOR = (80, 220, 80)
_BALL_COLOR = (0, 165, 255)
_HUD_COLOR = (0, 220, 0)


class PreviewCancelled(RuntimeError):
    pass


class ReselectRequested(RuntimeError):
    pass


def run_preview(
    grab_frame: Callable[[], np.ndarray],
    region: Region,
    detector: FootballDetector,
) -> None:
    """Capture, detect, and display frames until the user quits or reselects."""
    meter = _FpsMeter()
    cv2.namedWindow(_WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        while True:
            frame = grab_frame()
            result = detector.detect(frame)
            fps = meter.tick()
            view = frame.copy()
            _draw_detections(view, result.detections)
            _draw_hud(view, region, fps, result)
            cv2.imshow(_WINDOW_NAME, view)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                raise PreviewCancelled("Onizleme kapatildi.")
            if key == ord("r"):
                raise ReselectRequested()
    finally:
        cv2.destroyWindow(_WINDOW_NAME)
        cv2.waitKey(1)


def _draw_detections(frame: np.ndarray, detections: tuple[Detection, ...]) -> None:
    for item in detections:
        color = _BALL_COLOR if item.label == "ball" else _PLAYER_COLOR
        cv2.rectangle(frame, (item.x1, item.y1), (item.x2, item.y2), color, 2)
        caption = f"{item.label} {item.confidence:.2f}"
        text_origin = (item.x1, max(16, item.y1 - 6))
        cv2.putText(
            frame,
            caption,
            text_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )


def _draw_hud(
    frame: np.ndarray,
    region: Region,
    fps: float,
    result: DetectionResult,
) -> None:
    players, balls = count_labels(result.detections)
    lines = [
        f"Det FPS: {fps:.1f}  Infer: {result.latency_ms:.0f}ms  Device: {result.device}",
        f"Players: {players}  Ball: {balls}  {region.width}x{region.height}",
        "q/ESC: cikis  |  r: yeniden bolge sec",
    ]
    y = 28
    for line in lines:
        cv2.putText(
            frame,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            _HUD_COLOR,
            2,
            cv2.LINE_AA,
        )
        y += 26


class _FpsMeter:
    def __init__(self, window: int = _FPS_WINDOW) -> None:
        self._times: deque[float] = deque(maxlen=window)

    def tick(self) -> float:
        now = time.perf_counter()
        self._times.append(now)
        if len(self._times) < 2:
            return 0.0
        elapsed = self._times[-1] - self._times[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._times) - 1) / elapsed
