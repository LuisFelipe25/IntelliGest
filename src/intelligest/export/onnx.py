from __future__ import annotations

import sys
from pathlib import Path

from intelligest.training.yolov5 import require_yolov5


def build_export_command(weights: Path, image_size: int = 224) -> list[str]:
    yolo = require_yolov5()
    return [
        sys.executable,
        str(yolo / "export.py"),
        "--weights",
        str(weights),
        "--imgsz",
        str(image_size),
        "--include",
        "onnx",
    ]
