from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from intelligest.config import DatasetProfile, ToolchainConfig, project_root


def require_yolov5(root: Path | None = None) -> Path:
    config = ToolchainConfig.load(root)
    required = (
        config.yolov5_path / "classify" / "train.py",
        config.yolov5_path / "classify" / "val.py",
        config.yolov5_path / "export.py",
        config.yolov5_path / "models" / "yolo.py",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("La integración YOLOv5 está incompleta: " + ", ".join(missing))
    if config.source_commit and (config.yolov5_path / ".git").is_dir():
        actual = subprocess.check_output(
            ["git", "-C", str(config.yolov5_path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if actual != config.source_commit:
            raise RuntimeError(
                f"El checkout YOLOv5/YARVIS debe estar en {config.source_commit}; se encontró {actual}"
            )
    return config.yolov5_path


def build_train_command(
    profile: DatasetProfile,
    dataset: Path,
    model: str,
    epochs: int,
    batch_size: int,
    image_size: int,
    device: str,
    seed: int,
    output: Path,
    root: Path | None = None,
) -> list[str]:
    yolo = require_yolov5(root)
    return [
        sys.executable,
        str(yolo / "classify" / "train.py"),
        "--model",
        model,
        "--data",
        str(dataset),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--imgsz",
        str(image_size),
        "--device",
        device,
        "--seed",
        str(seed),
        "--project",
        str(output),
        "--name",
        profile.id,
        "--exist-ok",
    ]


def build_evaluate_command(weights: Path, dataset: Path, image_size: int, device: str) -> list[str]:
    yolo = require_yolov5()
    return [
        sys.executable,
        str(yolo / "classify" / "val.py"),
        "--weights",
        str(weights),
        "--data",
        str(dataset),
        "--imgsz",
        str(image_size),
        "--device",
        device,
    ]


def write_experiment_record(
    path: Path,
    profile: DatasetProfile,
    dataset: Path,
    command: list[str],
) -> None:
    root = project_root()
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except subprocess.CalledProcessError:
        commit = "uncommitted"
    toolchain = ToolchainConfig.load(root)
    record = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "profile": profile.id,
        "classes": list(profile.classes),
        "dataset": str(dataset),
        "code_commit": commit,
        "yolov5_source_commit": toolchain.source_commit,
        "yolov5_upstream_base": toolchain.upstream_base_commit,
        "command": command,
        "environment": {"python": sys.version, "platform": platform.platform()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run(command: list[str], execute: bool) -> int:
    print(subprocess.list2cmdline(command))
    if not execute:
        return 0
    return subprocess.run(command, check=False).returncode

