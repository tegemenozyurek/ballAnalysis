from __future__ import annotations

import cv2
import numpy as np

from live_football_vision.region import Region

_WINDOW_NAME = "Live Football Vision — Bolge Sec"
_MAX_DISPLAY_WIDTH = 1600
_MAX_DISPLAY_HEIGHT = 900
_MIN_REGION_SIZE = 32


class RegionSelectCancelled(RuntimeError):
    pass


def select_region(
    screenshot: np.ndarray,
    screen: Region,
    pixel_scale: float,
) -> Region:
    """Let the user drag-select a rectangle on a screenshot.

    Coordinates are mapped back to mss logical screen space, including Retina.
    """
    display, display_scale = _fit_for_display(screenshot)
    selector = _DragSelector()

    cv2.namedWindow(_WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(_WINDOW_NAME, selector.on_mouse)

    try:
        while True:
            view = display.copy()
            _draw_instructions(view)
            if selector.start is not None and selector.end is not None:
                cv2.rectangle(view, selector.start, selector.end, (0, 220, 0), 2)
            cv2.imshow(_WINDOW_NAME, view)

            key = cv2.waitKey(16) & 0xFF
            if key in (27, ord("q")):
                raise RegionSelectCancelled("Bolge secimi iptal edildi.")
            if key in (13, 32):  # Enter or Space
                box = selector.normalized_box()
                if box is None:
                    continue
                x1, y1, x2, y2 = box
                if (x2 - x1) < _MIN_REGION_SIZE or (y2 - y1) < _MIN_REGION_SIZE:
                    continue
                return _map_display_box_to_region(
                    x1, y1, x2, y2, screen, display_scale, pixel_scale
                )
    finally:
        cv2.destroyWindow(_WINDOW_NAME)
        cv2.waitKey(1)


def _fit_for_display(image: np.ndarray) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    scale = min(_MAX_DISPLAY_WIDTH / width, _MAX_DISPLAY_HEIGHT / height, 1.0)
    if scale == 1.0:
        return image, 1.0
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA), scale


def _map_display_box_to_region(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    screen: Region,
    display_scale: float,
    pixel_scale: float,
) -> Region:
    scale = display_scale * pixel_scale
    left = screen.left + int(round(x1 / scale))
    top = screen.top + int(round(y1 / scale))
    width = max(1, int(round((x2 - x1) / scale)))
    height = max(1, int(round((y2 - y1) / scale)))
    return Region(left=left, top=top, width=width, height=height)


def _draw_instructions(view: np.ndarray) -> None:
    text = "Sol tikla ve surukle | ENTER: onay | ESC: iptal"
    cv2.putText(
        view,
        text,
        (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 220, 0),
        2,
        cv2.LINE_AA,
    )


class _DragSelector:
    def __init__(self) -> None:
        self.start: tuple[int, int] | None = None
        self.end: tuple[int, int] | None = None
        self._dragging = False

    def on_mouse(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self._dragging = True
            self.start = (x, y)
            self.end = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self._dragging:
            self.end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self._dragging = False
            self.end = (x, y)

    def normalized_box(self) -> tuple[int, int, int, int] | None:
        if self.start is None or self.end is None:
            return None
        x1, y1 = self.start
        x2, y2 = self.end
        left, right = sorted((x1, x2))
        top, bottom = sorted((y1, y2))
        if right == left or bottom == top:
            return None
        return left, top, right, bottom
