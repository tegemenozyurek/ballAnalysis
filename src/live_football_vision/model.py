from __future__ import annotations

YOLO_WEIGHTS = "yolo11n.pt"
PLAYER_CLASS_ID = 0
BALL_CLASS_ID = 32
DETECT_CLASS_IDS = (PLAYER_CLASS_ID, BALL_CLASS_ID)
CLASS_LABELS = {
    PLAYER_CLASS_ID: "player",
    BALL_CLASS_ID: "ball",
}


def resolve_device() -> str:
    import torch

    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_yolo_model(weights: str = YOLO_WEIGHTS, device: str | None = None):
    """Load a COCO YOLO model on the best local device (MPS on Apple Silicon)."""
    from ultralytics import YOLO

    selected = device or resolve_device()
    model = YOLO(weights)
    try:
        model.fuse()
    except Exception:
        pass
    return model, selected
