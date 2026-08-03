from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from intelligest.config import ModelContract, project_root
from intelligest.inference.engine import ONNXEngine
from intelligest.integrations.udp import UDPActionConfig, send_action


def build_parser() -> argparse.ArgumentParser:
    root = project_root()
    parser = argparse.ArgumentParser(description="IntelliGest desktop and headless inference")
    parser.add_argument("--contract", type=Path, default=root / "configs/models/arm_poses_7_app.json")
    parser.add_argument("--actions", type=Path, default=root / "configs/actions/arm_poses_7.json")
    parser.add_argument("--model", type=Path, help="Override the external model path")
    parser.add_argument("--provider", choices=["CPU", "CUDA", "DirectML"], default="CPU")
    parser.add_argument("--source", default="0", help="Camera index, image or video path")
    parser.add_argument("--headless", action="store_true", help="Run one image inference without PySide6")
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="Validate configuration without loading a model",
    )
    parser.add_argument("--no-udp", action="store_true", help="Disable network actions")
    return parser


def _headless(engine: ONNXEngine, source: str) -> int:
    path = Path(source)
    if not path.is_file():
        raise ValueError("El modo headless requiere una ruta de imagen existente en --source")
    prediction = engine.predict_image(path)
    print(
        json.dumps(
            {
                "class": prediction.class_name,
                "confidence": prediction.confidence,
                "probabilities": dict(zip(engine.contract.classes, prediction.probabilities, strict=True)),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _run_gui(engine: ONNXEngine, actions: UDPActionConfig, source: str, enable_udp: bool) -> int:
    try:
        import cv2
        from PySide6.QtCore import Qt, QThread, QTimer, Signal
        from PySide6.QtGui import QImage, QPixmap
        from PySide6.QtWidgets import (
            QApplication,
            QHBoxLayout,
            QLabel,
            QMessageBox,
            QProgressBar,
            QPushButton,
            QSlider,
            QVBoxLayout,
            QWidget,
        )
    except ImportError as exc:
        raise RuntimeError("Instala IntelliGest con el extra desktop para abrir la interfaz") from exc

    class CaptureWorker(QThread):
        frame_ready = Signal(object)
        prediction_ready = Signal(object, float)
        failed = Signal(str)

        def __init__(self):
            super().__init__()
            self.running = False

        def run(self):
            capture_source = int(source) if source.isdigit() else source
            if isinstance(capture_source, str) and Path(capture_source).suffix.lower() in {
                ".bmp",
                ".jpeg",
                ".jpg",
                ".png",
                ".webp",
            }:
                frame = cv2.imread(capture_source)
                if frame is None:
                    self.failed.emit(f"No se pudo leer la imagen: {source}")
                    return
                self.frame_ready.emit(frame)
                self.prediction_ready.emit(engine.predict_bgr(frame), 0.0)
                return
            backend = (
                cv2.CAP_DSHOW
                if sys.platform.startswith("win") and isinstance(capture_source, int)
                else 0
            )
            capture = cv2.VideoCapture(capture_source, backend)
            if not capture.isOpened():
                self.failed.emit(f"No se pudo abrir la fuente: {source}")
                return
            self.running = True
            previous = time.monotonic()
            try:
                while self.running:
                    ok, frame = capture.read()
                    if not ok:
                        if not isinstance(capture_source, int):
                            break
                        continue
                    prediction = engine.predict_bgr(frame)
                    now = time.monotonic()
                    fps = 1.0 / max(now - previous, 1e-6)
                    previous = now
                    self.frame_ready.emit(frame)
                    self.prediction_ready.emit(prediction, fps)
            except Exception as exc:
                self.failed.emit(str(exc))
            finally:
                capture.release()

        def stop(self):
            self.running = False
            self.wait(1000)

    class Window(QWidget):
        def __init__(self):
            super().__init__()
            self.setWindowTitle(f"IntelliGest — {engine.contract.profile}")
            self.setMinimumSize(1024, 620)
            self.worker = None
            self.last_frame = None
            self.candidate = None
            self.candidate_since = None
            self.last_sent = None

            self.video = QLabel("Fuente detenida")
            self.video.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.video.setMinimumSize(640, 480)
            self.heading = QLabel("Sin predicción")
            self.heading.setStyleSheet("font-size: 22px; font-weight: 600")
            self.fps = QLabel("0 FPS")
            self.bars = {}
            panel = QVBoxLayout()
            panel.addWidget(self.heading)
            panel.addWidget(self.fps)
            for class_name in engine.contract.classes:
                label = QLabel(class_name)
                bar = QProgressBar()
                bar.setRange(0, 1000)
                bar.setFormat("%p%")
                panel.addWidget(label)
                panel.addWidget(bar)
                self.bars[class_name] = bar
            self.threshold = QSlider(Qt.Orientation.Horizontal)
            self.threshold.setRange(0, 100)
            self.threshold.setValue(round(actions.minimum_confidence * 100))
            panel.addWidget(QLabel("Umbral de confianza"))
            panel.addWidget(self.threshold)
            start = QPushButton("Iniciar")
            stop = QPushButton("Detener")
            snapshot = QPushButton("Capturar frame")
            start.clicked.connect(self.start)
            stop.clicked.connect(self.stop)
            snapshot.clicked.connect(self.snapshot)
            panel.addWidget(start)
            panel.addWidget(stop)
            panel.addWidget(snapshot)
            panel.addStretch()
            layout = QHBoxLayout(self)
            layout.addWidget(self.video, 3)
            right = QWidget()
            right.setLayout(panel)
            layout.addWidget(right, 2)
            QTimer.singleShot(0, self.start)

        def start(self):
            if self.worker and self.worker.isRunning():
                return
            self.worker = CaptureWorker()
            self.worker.frame_ready.connect(self.update_frame)
            self.worker.prediction_ready.connect(self.update_prediction)
            self.worker.failed.connect(self.error)
            self.worker.start()

        def stop(self):
            if self.worker:
                self.worker.stop()

        def snapshot(self):
            if self.last_frame is None:
                return
            destination = project_root() / "reports" / "generated" / f"capture-{int(time.time())}.jpg"
            destination.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(destination), self.last_frame)
            self.heading.setText(f"Captura: {destination.name}")

        def update_frame(self, frame):
            self.last_frame = frame.copy()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channels = rgb.shape
            image = QImage(rgb.data, width, height, channels * width, QImage.Format.Format_RGB888).copy()
            self.video.setPixmap(
                QPixmap.fromImage(image).scaled(
                    self.video.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )

        def update_prediction(self, prediction, fps):
            threshold = self.threshold.value() / 100.0
            self.heading.setText(f"{prediction.class_name} — {prediction.confidence:.1%}")
            self.fps.setText(f"{fps:.1f} FPS")
            for name, probability in zip(engine.contract.classes, prediction.probabilities, strict=True):
                self.bars[name].setValue(round(probability * 1000))
            now = time.monotonic()
            if prediction.class_name != self.candidate or prediction.confidence < threshold:
                self.candidate = prediction.class_name
                self.candidate_since = now
                return
            if self.candidate_since is None or now - self.candidate_since < actions.minimum_stable_seconds:
                return
            if self.last_sent == prediction.class_name:
                return
            if enable_udp:
                send_action(actions, prediction.class_name)
            self.last_sent = prediction.class_name

        def error(self, message):
            QMessageBox.critical(self, "IntelliGest", message)

        def closeEvent(self, event):
            self.stop()
            event.accept()

    app = QApplication(sys.argv)
    window = Window()
    window.show()
    return app.exec()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    contract = ModelContract.load(args.contract)
    actions = UDPActionConfig.load(args.actions)
    if set(actions.class_payloads) != set(contract.classes):
        raise ValueError("Las acciones configuradas no coinciden con las clases del contrato")
    if args.check_config:
        print(
            json.dumps(
                {
                    "profile": contract.profile,
                    "classes": list(contract.classes),
                    "model": str(args.model or contract.path) if (args.model or contract.path) else None,
                    "udp": {"host": actions.host, "port": actions.port, "enabled": not args.no_udp},
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    engine = ONNXEngine(contract, args.provider, args.model)
    if args.headless:
        return _headless(engine, args.source)
    return _run_gui(engine, actions, args.source, not args.no_udp)


if __name__ == "__main__":
    raise SystemExit(main())
