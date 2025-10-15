#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import sys
import argparse
import random
import numpy as np
from pathlib import Path

TARGET_DEFAULT = 5000
RANDOM_SEED = 123

# ---------------- utilidades ----------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def estimate_total_frames(cap):
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total > 0:
        return total
    # Fallback (raro): contar uno a uno
    pos = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        pos += 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return pos

def sample_indices(total_frames, target):
    if total_frames <= 0:
        return []
    if target >= total_frames:
        return list(range(total_frames))
    step = (total_frames - 1) / (target - 1)
    idxs = [int(round(i * step)) for i in range(target)]
    idxs = sorted(set(max(0, min(total_frames - 1, i)) for i in idxs))
    while len(idxs) < target:
        idxs.append(idxs[-1])
    return idxs[:target]

def save_image(img, out_path, ext, quality):
    params = []
    e = ext.lower()
    if e in {"jpg", "jpeg"}:
        params = [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
    elif e == "webp":
        params = [cv2.IMWRITE_WEBP_QUALITY, int(quality)]
    ensure_dir(out_path.parent)
    return cv2.imwrite(str(out_path), img, params)

# ---------------- aumentaciones suaves ----------------
def augment_frame(img):
    out = img.copy()
    # flip horizontal (50%)
    if random.random() < 0.5:
        out = cv2.flip(out, 1)

    # brillo/contraste pequeños
    alpha = 1.0 + random.uniform(-0.1, 0.1)  # contraste
    beta = random.uniform(-12, 12)           # brillo
    out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)

    # ligera rotación/traslación (50%)
    if random.random() < 0.5:
        h, w = out.shape[:2]
        angle = random.uniform(-3.0, 3.0)
        tx = random.uniform(-0.01 * w, 0.01 * w)
        ty = random.uniform(-0.01 * h, 0.01 * h)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        M[:, 2] += (tx, ty)
        out = cv2.warpAffine(
            out, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
        )

    # ruido gaussiano leve (50%)
    if random.random() < 0.5:
        noise = np.random.normal(0, 4.0, out.shape).astype(np.float32)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return out

# ---------------- pipeline principal ----------------
def extract_5k_resize(
    video_path: Path,
    out_root: Path,
    prefix: str,
    target_count: int = TARGET_DEFAULT,
    width: int = 224,
    height: int = 224,
    image_format: str = "jpg",
    quality: int = 95,
    do_augment_if_short: bool = True,
):
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir el video: {video_path}", file=sys.stderr)
        return 0

    total_frames = estimate_total_frames(cap)

    # Guardar dentro de: output/prefix/
    out_dir = out_root / prefix
    ensure_dir(out_dir)

    # Decisión automática
    target_to_extract = min(total_frames, target_count)
    idxs = sample_indices(total_frames, target_to_extract)
    idxs = sorted(idxs)

    # iterador de objetivos
    next_goal_iter = iter(idxs)
    try:
        next_goal = next(next_goal_iter)
    except StopIteration:
        next_goal = None

    saved_count = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if next_goal is not None and frame_idx == next_goal:
            # --- redimensionar EXACTO a WxH (sin letterbox, puede estirar) ---
            resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            fname = f"{prefix}_{saved_count}.{image_format}"  # nombre_prefijo_indice
            out_path = out_dir / fname
            if save_image(resized, out_path, image_format, quality):
                saved_count += 1
            try:
                next_goal = next(next_goal_iter)
            except StopIteration:
                next_goal = None
                if saved_count >= target_to_extract:
                    break

        frame_idx += 1

    cap.release()

    # Completar con aumentaciones si faltan imágenes
    if do_augment_if_short and saved_count < target_count and saved_count > 0:
        # leer de disco lo ya guardado
        originals = []
        # Intentar cargar hasta saved_count imágenes para variar
        for i in range(saved_count):
            p = out_dir / f"{prefix}_{i}.{image_format}"
            im = cv2.imread(str(p))
            if im is not None:
                originals.append(im)

        needed = target_count - saved_count
        aug_count = 0
        while aug_count < needed and originals:
            src = random.choice(originals)
            aug = augment_frame(src)
            # asegurar tamaño exacto
            aug = cv2.resize(aug, (width, height), interpolation=cv2.INTER_AREA)
            fname = f"{prefix}_{saved_count}.{image_format}"
            out_path = out_dir / fname
            if save_image(aug, out_path, image_format, quality):
                saved_count += 1
                aug_count += 1
        print(f"[INFO] Aumentaciones generadas: {aug_count}")

    print(f"[OK] {video_path.name}: {saved_count} imágenes guardadas en {out_dir}")
    if saved_count < target_count:
        print(f"[WARN] El video no alcanzó {target_count}; total={saved_count} (incluye aumentaciones si aplican).")
    return saved_count

# ---------------- CLI ----------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Extrae N imágenes (o completa con aumentaciones si no alcanza) y redimensiona EXACTO a WxH sin letterbox. Nombres: <prefijo>_0.jpg, <prefijo>_1.jpg, ..."
    )
    p.add_argument("video", type=str, help="Ruta al video de entrada.")
    p.add_argument("--width", type=int, default=224, help="Ancho destino (px).")
    p.add_argument("--height", type=int, default=224, help="Alto destino (px).")
    p.add_argument("--target", type=int, default=TARGET_DEFAULT, help="Cantidad objetivo (por defecto 5000).")
    p.add_argument("--format", type=str, default="jpg", help="Formato: jpg|png|bmp|webp.")
    p.add_argument("--quality", type=int, default=95, help="Calidad JPG/WEBP (1-100).")
    p.add_argument("-o", "--output", type=str, default="dataset_frames", help="Carpeta raíz de salida.")
    p.add_argument("--no-augment", action="store_true", help="No completar con aumentaciones si faltan imágenes.")
    p.add_argument("prefix", type=str, help="Prefijo de nombre de archivo. Ej: 'mi_nombre' genera mi_nombre_0.jpg, mi_nombre_1.jpg, ...")
    return p.parse_args()

def main():
    args = parse_args()

    video = Path(args.video)
    if not video.exists():
        print(f"[ERROR] No existe el video: {video}", file=sys.stderr)
        sys.exit(1)

    out_root = Path(args.output)
    ensure_dir(out_root)

    extract_5k_resize(
        video_path=video,
        out_root=out_root,
        prefix=args.prefix,
        target_count=int(args.target),
        width=int(args.width),
        height=int(args.height),
        image_format=args.format.lower(),
        quality=int(args.quality),
        do_augment_if_short=not args.no_augment,
    )

if __name__ == "__main__":
    main()
