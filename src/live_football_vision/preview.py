from __future__ import annotations

from collections import deque
from collections.abc import Callable
import time

import cv2
import numpy as np

from live_football_vision.region import Region

_WINDOW_NAME = "Live Football Vision — Onizleme"
_FPS_WINDOW = 30


class PreviewCancelled(RuntimeError):
    pass


class ReselectRequested(RuntimeError):
    pass


def run_preview(
    grab_frame: Callable[[], np.ndarray],
    region: Region,
) -> None:
    """Show captured frames until the user quits or asks to reselect."""
    meter = _FpsMeter()
    cv2.namedWindow(_WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        while True:
            frame = grab_frame()
            fps = meter.tick()
            view = frame.copy()
            _draw_hud(view, region, fps)
            cv2.imshow(_WINDOW_NAME, view)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                raise PreviewCancelled("Onizleme kapatildi.")
            if key == ord("r"):
                raise ReselectRequested()
    finally:
        cv2.destroyWindow(_WINDOW_NAME)
        cv2.waitKey(1)


def _draw_hud(frame: np.ndarray, region: Region, fps: float) -> None:
    lines = [
        f"FPS: {fps:.1f}",
        f"Bolge: {region.width}x{region.height} @ ({region.left},{region.top})",
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
            (0, 220, 0),
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
