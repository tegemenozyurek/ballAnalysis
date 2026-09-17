from __future__ import annotations

import argparse
import sys

from live_football_vision.capture import (
    ScreenCapture,
    ScreenCaptureError,
    is_likely_permission_blocked,
)
from live_football_vision.detect import FootballDetector
from live_football_vision.model import YOLO_WEIGHTS, load_yolo_model
from live_football_vision.preview import PreviewCancelled, ReselectRequested, run_preview
from live_football_vision.region import Region
from live_football_vision.region_select import RegionSelectCancelled, select_region

_PERMISSION_HELP = """
macOS Ekran Kaydi izni gerekli.

Sistem Ayarlari > Gizlilik ve Guvenlik > Ekran Kaydi
(veya Ekran ve Sistem Sesi Kaydi)

Bu uygulamayi calistirdigin uygulama icin izni ac:
  - Terminal.app
  - iTerm
  - Cursor
Sonra uygulamayi tamamen kapatip yeniden dene.
""".strip()


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        detector = _load_detector()
        with ScreenCapture() as capture:
            _ensure_capture_allowed(capture)
            region = args.region
            while True:
                if region is None:
                    region = _pick_region(capture)
                    print(f"Secilen bolge: {region}")
                try:
                    run_preview(lambda: capture.grab(region), region, detector)
                except ReselectRequested:
                    region = None
                    continue
                except PreviewCancelled:
                    print("Onizleme kapatildi.")
                    return 0
    except (ScreenCaptureError, RegionSelectCancelled) as exc:
        print(exc, file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nDurduruldu.")
        return 0


def _load_detector() -> FootballDetector:
    print(f"Model yukleniyor: {YOLO_WEIGHTS}")
    model, device = load_yolo_model()
    detector = FootballDetector(model, device)
    print(f"Device: {device}")
    detector.warmup()
    print("Model hazir.")
    return detector


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="lfv",
        description="Mac ekranindan bir bolge yakala, oyuncu ve topu gercek zamanli tespit et.",
    )
    parser.add_argument(
        "--region",
        type=_parse_region,
        default=None,
        help="left,top,width,height (ornek: 100,80,1280,720). Verilmezse surukleyerek secilir.",
    )
    return parser.parse_args(argv)


def _parse_region(value: str) -> Region:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("region formati: left,top,width,height")
    try:
        left, top, width, height = (int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("region degerleri tam sayi olmali") from exc
    try:
        return Region(left=left, top=top, width=width, height=height)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _ensure_capture_allowed(capture: ScreenCapture) -> None:
    frame, _, _ = capture.grab_virtual_screen()
    if is_likely_permission_blocked(frame):
        raise ScreenCaptureError(_PERMISSION_HELP)


def _pick_region(capture: ScreenCapture) -> Region:
    frame, screen, pixel_scale = capture.grab_virtual_screen()
    return select_region(frame, screen, pixel_scale)


if __name__ == "__main__":
    raise SystemExit(main())
