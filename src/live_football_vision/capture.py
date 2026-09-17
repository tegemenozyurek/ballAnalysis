from __future__ import annotations

import mss
import numpy as np

from live_football_vision.region import Region

_BLACK_FRAME_MEAN_THRESHOLD = 1.0


class ScreenCaptureError(RuntimeError):
    pass


class ScreenCapture:
    """Grabs a macOS screen region as a BGR numpy frame."""

    def __init__(self) -> None:
        self._sct = mss.mss()

    def close(self) -> None:
        self._sct.close()

    def __enter__(self) -> ScreenCapture:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def virtual_screen(self) -> Region:
        monitor = self._sct.monitors[0]
        return Region(
            left=int(monitor["left"]),
            top=int(monitor["top"]),
            width=int(monitor["width"]),
            height=int(monitor["height"]),
        )

    def grab(self, region: Region) -> np.ndarray:
        try:
            raw = self._sct.grab(region.to_mss())
        except Exception as exc:
            raise ScreenCaptureError(
                "Ekran yakalama basarisiz. macOS Ekran Kaydi iznini kontrol edin."
            ) from exc

        frame = np.asarray(raw)
        if frame.size == 0:
            raise ScreenCaptureError("Bos kare alindi; bolge gecersiz olabilir.")
        return _to_bgr(frame)

    def grab_virtual_screen(self) -> tuple[np.ndarray, Region, float]:
        """Return (BGR frame, logical region, retina pixel scale)."""
        region = self.virtual_screen()
        raw = self._sct.grab(region.to_mss())
        frame = _to_bgr(np.asarray(raw))
        pixel_scale = raw.width / region.width if region.width else 1.0
        return frame, region, float(pixel_scale)


def _to_bgr(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 3 and frame.shape[2] == 4:
        return np.ascontiguousarray(frame[:, :, :3])
    return frame


def is_likely_permission_blocked(frame: np.ndarray) -> bool:
    """Denied Screen Recording on macOS often yields a near-black frame."""
    return float(frame.mean()) < _BLACK_FRAME_MEAN_THRESHOLD
