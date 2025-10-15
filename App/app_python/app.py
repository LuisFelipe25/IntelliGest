#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YARVIS UI — Interfaz bonita con PySide6
- Cámara a la izquierda, panel de detección a la derecha
- Carga modelo ONNX automáticamente desde "modelo.onnx"
- Toma los labels exclusivamente desde "labels.txt" (una línea por clase, en ese orden)
- Muestra Top-1 + barras de probabilidad
- Botones: Iniciar/Detener, Capturar frame, Cambiar umbral
- Envia por UDP (broadcast) una letra (a..z) cuando el Top-1 se mantiene ≥ min_stable s

Requisitos:
  pip install PySide6 onnxruntime opencv-python numpy

Ejemplo (arranca solo):
  python app.py  # usa modelo.onnx y labels.txt por defecto

Opcionales:
  --labels-txt labels.txt
  --imgsz 224 --camera 0 --topk 3 --thres 0.4
  --min-stable 3.0 --ip 255.255.255.255 --port 1097 --quiet
"""

from __future__ import annotations
import sys, time
from pathlib import Path
import argparse
from typing import List, Tuple, Optional

import numpy as np
import cv2
import onnxruntime as ort

from PySide6.QtCore import Qt, QThread, Signal, QSize, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QSlider, QSplitter, QFileDialog, QComboBox, QProgressBar, QScrollArea,
    QMessageBox, QSpinBox, QSizePolicy
)

# --- Emisor UDP (acciones.py debe estar al lado de este archivo) ---
try:
    import acciones  # debe exponer acciones.send_udp(letter, ip, port)
except Exception as e:
    acciones = None
    print(f"[WARN] acciones.py no disponible: {e}", file=sys.stderr)


# -------------------- CLI --------------------
def parse_args():
    ap = argparse.ArgumentParser(description="YARVIS ONNX UI (webcam)")
    ap.add_argument("--model", type=str, default="modelo.onnx", help="Ruta del modelo ONNX (se carga automáticamente)")
    ap.add_argument("--labels-txt", type=str, default="labels.txt", help="Archivo .txt con una clase por línea (orden exacto)")
    ap.add_argument("--provider", type=str, default="CPU", choices=["CPU", "CUDA", "DirectML"], help="Proveedor ONNX Runtime")
    ap.add_argument("--imgsz", type=int, default=224, help="Tamaño cuadrado de entrada")
    ap.add_argument("--camera", type=int, default=0, help="Índice de la webcam (0 por defecto)")
    ap.add_argument("--topk", type=int, default=1, help="Top-K a mostrar en el encabezado")
    ap.add_argument("--thres", type=float, default=0.4, help="Umbral para color verde/naranja del encabezado")
    # NUEVOS (envío UDP cuando Top-1 estable)
    ap.add_argument("--min-stable", type=float, default=3.0, help="Segundos de estabilidad requeridos para enviar (default 3.0)")
    ap.add_argument("--ip", type=str, default="255.255.255.255", help="IP destino (default broadcast)")
    ap.add_argument("--port", type=int, default=1097, help="Puerto destino (default 1097)")
    ap.add_argument("--quiet", action="store_true", help="Oculta logs de estado/envío")
    return ap.parse_args()


# -------------------- ONNX helpers --------------------
def load_session(model_path: str, provider: str):
    if provider == "DirectML":
        providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
    elif provider == "CUDA":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    sess = ort.InferenceSession(model_path, providers=providers)
    inp = sess.get_inputs()[0]
    out = sess.get_outputs()[0]
    return sess, inp.name, inp.shape, out.name


def infer_shape_layout(inp_shape, imgsz) -> Tuple[str, List[int]]:
    shape = list(inp_shape)
    if len(shape) != 4:
        raise ValueError(f"Esperaba input 4D, recibido: {shape}")
    if shape[1] in (1, 3) or shape[1] is None:
        layout = "NCHW"
        shape[0] = 1
        shape[2] = imgsz
        shape[3] = imgsz
    elif shape[-1] in (1, 3) or shape[-1] is None:
        layout = "NHWC"
        shape[0] = 1
        shape[1] = imgsz
        shape[2] = imgsz
        shape[3] = 3
    else:
        if shape[1] in (1, 3):
            layout = "NCHW"
        elif shape[-1] in (1, 3):
            layout = "NHWC"
        else:
            raise ValueError(f"No se pudo inferir layout desde {shape}")
    return layout, shape


def preprocess_bgr(frame_bgr: np.ndarray, imgsz: int, layout: str) -> np.ndarray:
    resized = cv2.resize(frame_bgr, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    x = rgb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = (x - mean) / std
    if layout == "NCHW":
        x = np.transpose(x, (2, 0, 1))
    x = np.expand_dims(x, axis=0).astype(np.float32)
    return x


def softmax(logits: np.ndarray) -> np.ndarray:
    v = logits - np.max(logits, axis=1, keepdims=True)
    e = np.exp(v)
    return e / np.sum(e, axis=1, keepdims=True)


def load_labels_from_args(args) -> List[str]:
    p = Path(args.labels_txt)
    if not p.exists():
        raise FileNotFoundError(f"No existe labels.txt: {p}")
    labels = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not labels:
        raise ValueError("labels.txt está vacío. Agrega una etiqueta por línea en el orden deseado.")
    if len(labels) > 26:
        raise ValueError("Hay más de 26 labels (mapa a..z).")
    return labels


def index_to_letter(idx: int) -> str:
    if not (0 <= idx < 26):
        raise ValueError("Índice fuera de rango para mapeo a..z.")
    return chr(ord('a') + idx)


# -------------------- Worker de captura e inferencia --------------------
class CameraWorker(QThread):
    frame_ready = Signal(np.ndarray)              # BGR frame para pintar
    probs_ready = Signal(np.ndarray, list, float) # probs, top_idxs, fps
    error = Signal(str)

    def __init__(self, cam_index: int, engine: "InferenceEngine"):
        super().__init__()
        self.cam_index = cam_index
        self.engine = engine
        self._running = False
        self._cap = None

    def run(self):
        try:
            backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else 0
            self._cap = cv2.VideoCapture(self.cam_index, backend)
            if not self._cap.isOpened():
                self.error.emit(f"No pude abrir la cámara {self.cam_index}")
                return
            self._running = True
            last_t = time.time()
            while self._running:
                ok, frame = self._cap.read()
                if not ok:
                    continue
                probs, top_idxs = self.engine.run(frame)
                now = time.time()
                fps = 1.0 / max(1e-6, now - last_t)
                last_t = now
                self.frame_ready.emit(frame)
                self.probs_ready.emit(probs, top_idxs.tolist(), fps)
        except Exception as e:
            self.error.emit(str(e))
        finally:
            if self._cap is not None:
                self._cap.release()

    def stop(self):
        self._running = False
        self.wait(500)


class InferenceEngine:
    def __init__(self, model_path: str, provider: str, imgsz: int, labels: List[str], topk: int):
        self.labels = labels
        self.topk = max(1, min(topk, len(labels)))
        self.sess, self.inp_name, inp_shape, self.out_name = load_session(model_path, provider)
        self.layout, _ = infer_shape_layout(inp_shape, imgsz)
        self.imgsz = imgsz

    def run(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x = preprocess_bgr(frame_bgr, self.imgsz, self.layout)
        y = self.sess.run([self.out_name], {self.inp_name: x})[0]
        y = y.reshape(1, -1)
        probs = softmax(y)[0]
        top_idxs = np.argsort(probs)[-self.topk:][::-1]
        return probs, top_idxs


# -------------------- UI principal --------------------
class YARVISWindow(QWidget):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.labels = load_labels_from_args(args)
        model_file = Path(args.model)
        if not model_file.exists():
            QMessageBox.critical(self, "Error", f"No se encontró el modelo: {model_file.resolve()}")
            sys.exit(1)

        self.engine = InferenceEngine(str(model_file), args.provider, args.imgsz, self.labels, args.topk)

        self.setWindowTitle("YARVIS — Pose Classifier")
        self.setMinimumSize(1024, 600)
        self._build_ui()

        self.worker: CameraWorker | None = None
        self._last_frame: np.ndarray | None = None

        # Estado para estabilidad/UDP
        self._cand_label: Optional[str] = None
        self._cand_since: Optional[float] = None
        self._last_sent_label: Optional[str] = None

        # Iniciar automáticamente la cámara e inferencia
        QTimer.singleShot(0, self.start_camera)

    # ---------- UI layout ----------
    def _build_ui(self):
        splitter = QSplitter(Qt.Horizontal)

        # Izquierda: video
        self.video_label = QLabel("Cámara apagada")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background:#0b1020;color:#9aa4b1;border-radius:16px;")
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        left_layout.addWidget(self.video_label)

        # Derecha: panel de predicción
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(12, 12, 12, 12)
        right_container.setStyleSheet(
            """
            QWidget { background:#0d1428; }
            QLabel { color:white; }
            QLabel#title { color:white; font-size:28px; font-weight:600; }
            QLabel#subtitle { color:#c2c8d0; font-size:13px; }
            QProgressBar { background:#141b33; border:0; height:20px; border-radius:10px; color:white; }
            QProgressBar::chunk { border-radius:10px; }
            QPushButton { background:#1a2240; color:white; border:0; padding:10px 14px; border-radius:12px; }
            QPushButton:hover { background:#233061; }
            QSlider::groove:horizontal { height:6px; background:#141b33; border-radius:3px; }
            QSlider::handle:horizontal { width:16px; background:white; border-radius:8px; margin:-6px 0; }
            QComboBox { background:#1a2240; color:white; padding:8px; border-radius:10px; }
            QSpinBox { color:white; }
            """
        )

        self.title = QLabel("Listo")
        self.title.setObjectName("title")
        self.subtitle = QLabel(f"Provider: {self.args.provider} · Input: {self.args.imgsz}×{self.args.imgsz}")
        self.subtitle.setObjectName("subtitle")

        # Controles
        controls = QHBoxLayout()
        self.btn_start = QPushButton("Iniciar")
        now = time.time()
        self.btn_stop = QPushButton("Detener")
        self.btn_snap = QPushButton("Capturar")
        self.btn_stop.setEnabled(False)
        controls.addWidget(self.btn_start)
        controls.addWidget(self.btn_stop)
        controls.addWidget(self.btn_snap)
        controls.addStretch(1)

        # Cámara index
        cam_box = QHBoxLayout()
        cam_lbl = QLabel("Cámara:")
        self.spin_cam = QSpinBox()
        self.spin_cam.setRange(0, 9)
        self.spin_cam.setValue(self.args.camera)
        cam_box.addWidget(cam_lbl)
        cam_box.addWidget(self.spin_cam)
        cam_box.addStretch(1)

        # Umbral
        th_box = QHBoxLayout()
        th_lbl = QLabel("Umbral:")
        self.slider_th = QSlider(Qt.Horizontal)
        self.slider_th.setRange(0, 100)
        self.slider_th.setValue(int(self.args.thres * 100))
        self.lbl_th_val = QLabel(f"{self.args.thres:.2f}")
        th_box.addWidget(th_lbl)
        th_box.addWidget(self.slider_th)
        th_box.addWidget(self.lbl_th_val)

        # FPS + Top-1
        self.lbl_fps = QLabel("FPS: —")
        self.lbl_top = QLabel("Top-1: —")
        self.lbl_fps.setObjectName("subtitle")
        self.lbl_top.setObjectName("subtitle")

        # Barras de probabilidad (scroll por si hay muchas clases)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.bars_container = QWidget()
        self.bars_layout = QVBoxLayout(self.bars_container)
        self.bar_widgets: List[Tuple[QLabel, QProgressBar]] = []
        for cls in self.labels:
            row = QHBoxLayout()
            lab = QLabel(cls)
            lab.setMinimumWidth(160)
            bar = QProgressBar()
            bar.setRange(0, 100)
            row.addWidget(lab)
            row.addWidget(bar)
            self.bars_layout.addLayout(row)
            self.bar_widgets.append((lab, bar))
        self.bars_layout.addStretch(1)
        self.scroll.setWidget(self.bars_container)

        # Ensamble derecha
        right_layout.addWidget(self.title)
        right_layout.addWidget(self.subtitle)
        right_layout.addLayout(controls)
        right_layout.addLayout(cam_box)
        right_layout.addLayout(th_box)
        right_layout.addWidget(self.lbl_top)
        right_layout.addWidget(self.lbl_fps)
        right_layout.addWidget(self.scroll, 1)

        # Splitter
        splitter.addWidget(left_container)
        splitter.addWidget(right_container)
        splitter.setSizes([700, 400])

        root = QVBoxLayout(self)
        root.addWidget(splitter)

        # Signals
        self.btn_start.clicked.connect(self.start_camera)
        self.btn_stop.clicked.connect(self.stop_camera)
        self.btn_snap.clicked.connect(self.save_snapshot)
        self.slider_th.valueChanged.connect(self._on_th_changed)

    # ---------- slots ----------
    def start_camera(self):
        if self.worker is not None:
            return
        cam_idx = int(self.spin_cam.value())
        self.worker = CameraWorker(cam_idx, self.engine)
        self.worker.frame_ready.connect(self.update_frame)
        self.worker.probs_ready.connect(self.update_probs)
        self.worker.error.connect(self._on_error)
        self.worker.start()
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.title.setText("Transmitiendo…")

    def stop_camera(self):
        if self.worker:
            self.worker.stop()
            self.worker = None
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.title.setText("Cámara apagada")

    def save_snapshot(self):
        if self._last_frame is None:
            return
        ts = int(time.time())
        path, _ = QFileDialog.getSaveFileName(self, "Guardar captura", f"capture_{ts}.jpg", "JPEG (*.jpg)")
        if path:
            cv2.imwrite(path, self._last_frame)

    def _on_th_changed(self, v: int):
        self.args.thres = v / 100.0
        self.lbl_th_val.setText(f"{self.args.thres:.2f}")

    def _on_error(self, msg: str):
        QMessageBox.critical(self, "Cámara", msg)
        self.stop_camera()

    # ---------- actualización visual ----------
    def update_frame(self, frame_bgr: np.ndarray):
        self._last_frame = frame_bgr.copy()
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        self.video_label.setPixmap(pix.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event):  # para que el video se reescale suave
        if self._last_frame is not None:
            self.update_frame(self._last_frame)
        super().resizeEvent(event)

    def update_probs(self, probs: np.ndarray, top_idxs: List[int], fps: float):
        # Barras
        for i, (_, bar) in enumerate(self.bar_widgets):
            p = int(round(float(probs[i]) * 100))
            bar.setValue(p)
            is_top = (i == top_idxs[0])
            bar.setStyleSheet(
                "QProgressBar { background:#141b33; border:0; height:20px; border-radius:10px; }"
                f"QProgressBar::chunk {{ background:{'#40c463' if is_top else '#2a9d8f'}; border-radius:10px; }}"
            )

        # Top-1 y encabezado
        best_idx = top_idxs[0]
        best_p = float(probs[best_idx])
        best_name = self.labels[best_idx]
        self.lbl_top.setText(f"Top-1: {best_name} — {best_p:.3f}")
        self.lbl_fps.setText(f"FPS: {fps:.1f}")

        color = "#40c463" if best_p >= self.args.thres else "#ff9f1a"
        self.title.setText(best_name.upper())
        self.title.setStyleSheet(f"color:{color}; font-size:32px; font-weight:700;")

        # ------- LÓGICA DE ESTABILIDAD + ENVÍO UDP -------
        now = time.monotonic()

        # Si cambió el candidato, reinicia temporizador **y** permite re-enviar si más tarde vuelve a estabilizarse
        if self._cand_label != best_name:
            self._cand_label = best_name
            self._cand_since = now
            self._last_sent_label = None  # <-- clave para poder reenviar la MISMA clase tras cambiar a otra y volver
            if not self.args.quiet:
                print(f"[STATE] Candidato='{self._cand_label}' (reinicia temporizador)", flush=True)
            return

        # Misma etiqueta: medir estabilidad
        if self._cand_since is None:
            self._cand_since = now
        stable_time = now - self._cand_since

        # ¿Alcanzó umbral?
        if stable_time >= self.args.min_stable:
            # Si ya la enviamos en este segmento de estabilidad, no repetir
            if self._last_sent_label == self._cand_label:
                return
            try:
                letter = index_to_letter(best_idx)
                if acciones is None:
                    if not self.args.quiet:
                        print(f"[WARN] acciones.py no disponible; no se envía UDP. ({self._cand_label} → {letter})", flush=True)
                else:
                    acciones.send_udp(letter=letter, ip=self.args.ip, port=self.args.port)
                    if not self.args.quiet:
                        print(f"[SEND] '{self._cand_label}' estable {stable_time:.2f}s → '{letter}' enviada a {self.args.ip}:{self.args.port}.", flush=True)
                self._last_sent_label = self._cand_label
            except Exception as e:
                print(f"[ERROR] Fallo enviando '{self._cand_label}': {e}", file=sys.stderr, flush=True)


def main():
    args = parse_args()
    app = QApplication(sys.argv)
    app.setApplicationDisplayName("YARVIS UI")
    win = YARVISWindow(args)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
