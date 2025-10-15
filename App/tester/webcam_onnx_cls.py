#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


# ---------- CLI ----------
def parse_args():
    ap = argparse.ArgumentParser(description="YARVIS ONNX runner (webcam / image / video)")
    ap.add_argument("--model", type=str, default="YARVIS.onnx", help="Ruta del modelo ONNX")

    # Labels por defecto (tal como pediste):
    # 0 → arms_down, 1 → both_arms_up, 2 → left_arm_out, 3 → right_arm_out
    ap.add_argument(
        "--labels",
        type=str,
        default="arms_crossed,arms_down,arms_side,arms_up,left_arm_side,left_arm_up,right_arm_side,right_arm_up",
        help="Clases separadas por comas (si no usas --labels-txt)",
    )
    ap.add_argument("--labels-txt", type=str, default="", help="Archivo .txt con una clase por línea")

    ap.add_argument(
        "--source",
        type=str,
        default="0",
        help="Índice de webcam (e.g., 0) o ruta a imagen/video (jpg/mp4/...)",
    )
    ap.add_argument("--imgsz", type=int, default=224, help="Tamaño cuadrado de entrada")
    ap.add_argument("--thres", type=float, default=0.4, help="Umbral para colorear el overlay")
    ap.add_argument("--provider", type=str, default="CPU", choices=["CPU", "CUDA", "DirectML"],
                    help="Proveedor ONNX Runtime (CPU/CUDA/DirectML)")
    ap.add_argument("--topk", type=int, default=1, help="Mostrar top-K (1 por defecto)")
    return ap.parse_args()


# ---------- ONNX helpers ----------
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


def infer_shape_layout(inp_shape, imgsz):
    """Devuelve (layout, shape_fixed) donde layout es 'NCHW' o 'NHWC'."""
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


# ---------- preprocess/postprocess ----------
def preprocess_bgr(frame_bgr: np.ndarray, imgsz: int, layout: str) -> np.ndarray:
    # Resize y BGR->RGB
    resized = cv2.resize(frame_bgr, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    x = rgb.astype(np.float32) / 255.0
    # Normalización tipo ImageNet
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = (x - mean) / std  # HWC
    if layout == "NCHW":
        x = np.transpose(x, (2, 0, 1))  # CHW
    x = np.expand_dims(x, axis=0).astype(np.float32)  # NCHW o NHWC
    return x


def softmax(logits: np.ndarray) -> np.ndarray:
    v = logits - np.max(logits, axis=1, keepdims=True)
    e = np.exp(v)
    return e / np.sum(e, axis=1, keepdims=True)


def load_labels(args):
    if args.labels_txt:
        p = Path(args.labels_txt)
        if not p.exists():
            raise FileNotFoundError(f"No existe labels.txt: {p}")
        labels = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    else:
        labels = [s.strip() for s in args.labels.split(",") if s.strip()]
    if not labels:
        raise ValueError("Debes pasar --labels a,b,c o --labels-txt labels.txt")
    return labels


# ---------- fuente (webcam/imagen/video) ----------
def open_source(src: str):
    # webcam si es dígito
    if src.isdigit():
        idx = int(src)
        if sys.platform.startswith("win"):
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)  # backend recomendado en Windows
        else:
            cap = cv2.VideoCapture(idx)
        return "webcam", cap
    # archivo
    p = Path(src)
    if not p.exists():
        raise FileNotFoundError(f"No existe la fuente: {src}")
    # imagen
    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        img = cv2.imread(str(p))
        if img is None:
            raise RuntimeError(f"No pude leer la imagen: {p}")
        return "image", img
    # video
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        raise RuntimeError(f"No pude abrir el video: {p}")
    return "video", cap


# ---------- main ----------
def main():
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[ERROR] No se encontró el modelo: {model_path.resolve()}")
        sys.exit(1)

    labels = load_labels(args)

    # Sesión y shape
    sess, inp_name, inp_shape, out_name = load_session(str(model_path), args.provider)
    layout, fixed = infer_shape_layout(inp_shape, args.imgsz)
    print(f"[INFO] Provider: {args.provider}")
    print(f"[INFO] Input: name={inp_name}, shape={inp_shape} -> {fixed}, layout={layout}")
    print(f"[INFO] Output: name={out_name}, num_classes={len(labels)}")

    # Fuente
    mode, source = open_source(args.source)

    def run_frame(frame):
        x = preprocess_bgr(frame, args.imgsz, layout)
        y = sess.run([out_name], {inp_name: x})[0]  # (1, C) normalmente
        y = y.reshape(1, -1)
        probs = softmax(y)[0]
        k = max(1, min(args.topk, len(probs)))
        top_idxs = np.argsort(probs)[-k:][::-1]
        return probs, top_idxs

    if mode == "image":
        frame = source
        probs, top_idxs = run_frame(frame)
        text = " | ".join([f"{labels[i]} {probs[i]:.3f}" for i in top_idxs])
        color = (0, 255, 0) if probs[top_idxs[0]] >= args.thres else (0, 165, 255)
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
        cv2.imshow("YARVIS ONNX", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # video / webcam
    cap = source
    last_t = time.time()
    print("[INFO] q: salir, s: guardar frame actual")
    count_saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            if mode == "video":
                break
            print("[WARN] Frame no leído, reintentando...")
            continue

        probs, top_idxs = run_frame(frame)
        now = time.time()
        fps = 1.0 / max(1e-6, (now - last_t))
        last_t = now

        text = " | ".join([f"{labels[i]} {probs[i]:.3f}" for i in top_idxs])
        color = (0, 255, 0) if probs[top_idxs[0]] >= args.thres else (0, 165, 255)
        cv2.putText(frame, text + f" | {fps:.1f} FPS", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

        cv2.imshow("YARVIS ONNX", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            out_path = Path(f"capture_{count_saved}.jpg")
            cv2.imwrite(str(out_path), frame)
            print(f"[INFO] Guardado {out_path.resolve()}")
            count_saved += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
