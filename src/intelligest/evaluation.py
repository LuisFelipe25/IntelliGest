from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from intelligest.config import project_root
from intelligest.inference.engine import ONNXEngine

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class EvalResults:
    accuracy: float
    confusion_matrix: list[list[int]]
    per_class_accuracy: dict[str, float]
    per_class_recall: dict[str, float]
    confusion_matrix_path: Path

    def to_dict(self) -> dict[str, object]:
        return {
            "accuracy": self.accuracy,
            "confusion_matrix": self.confusion_matrix,
            "per_class_accuracy": self.per_class_accuracy,
            "per_class_recall": self.per_class_recall,
            "confusion_matrix_path": str(self.confusion_matrix_path),
        }


def normalize_name(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def iter_image_files(root: Path) -> Iterable[Path]:
    for file_path in sorted(root.rglob("*")):
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS:
            yield file_path


def validate_test_structure(test_dir: Path, labels: tuple[str, ...]) -> dict[int, Path]:
    if not test_dir.exists() or not test_dir.is_dir():
        raise FileNotFoundError(f"El directorio de prueba no existe: {test_dir}")

    class_dirs = [path for path in sorted(test_dir.iterdir()) if path.is_dir()]
    if len(class_dirs) != len(labels):
        raise ValueError(
            f"El directorio de prueba debe contener {len(labels)} carpetas. "
            f"Se encontraron {len(class_dirs)} en {test_dir}."
        )

    folder_by_norm: dict[str, Path] = {}
    for class_dir in class_dirs:
        key = normalize_name(class_dir.name)
        if key in folder_by_norm:
            raise ValueError(f"Nombres duplicados de carpetas tras normalización: {class_dir.name}")
        folder_by_norm[key] = class_dir

    expected_norm = [normalize_name(label) for label in labels]
    missing = [labels[i] for i, key in enumerate(expected_norm) if key not in folder_by_norm]
    extras = [
        class_dir.name
        for class_dir in class_dirs
        if normalize_name(class_dir.name) not in set(expected_norm)
    ]

    if missing or extras:
        raise ValueError(
            f"Las carpetas de prueba no coinciden con las clases del perfil. "
            f"Faltan: {missing or 'ninguna'}. Inesperadas: {extras or 'ninguna'}."
        )

    return {idx: folder_by_norm[key] for idx, key in enumerate(expected_norm)}


def save_confusion_matrix(
    matrix_data: list[list[int]], labels: tuple[str, ...], output_path: Path
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Instala matplotlib y numpy para generar la matriz de confusión gráfica") from exc

    cm = np.array(matrix_data, dtype=np.int64)
    fig, ax = plt.subplots(figsize=(8.5, 7.0), dpi=300)
    image = ax.imshow(cm, cmap="Blues")
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Cantidad", rotation=-90, va="bottom")

    ax.set_title("Matriz de Confusión (Dataset de Prueba)")
    ax.set_xlabel("Etiqueta Predicha")
    ax.set_ylabel("Etiqueta Real")

    ticks = np.arange(len(labels))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticklabels(labels)

    threshold = cm.max() * 0.5 if cm.size and cm.max() > 0 else 0.0
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            value = int(cm[row, col])
            text_color = "white" if value > threshold else "black"
            ax.text(col, row, str(value), ha="center", va="center", color=text_color)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def evaluate_dataset(
    engine: ONNXEngine,
    test_dir: Path,
    output_image_path: Path | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> EvalResults:
    labels = engine.contract.classes
    folder_mapping = validate_test_structure(test_dir, labels)

    samples: list[tuple[Path, int]] = []
    for class_idx in range(len(labels)):
        class_folder = folder_mapping[class_idx]
        image_paths = list(iter_image_files(class_folder))
        if not image_paths:
            raise ValueError(f"No se encontraron imágenes en la carpeta de clase: {class_folder}")
        for image_path in image_paths:
            samples.append((image_path, class_idx))

    if not samples:
        raise ValueError(f"No hay imágenes de prueba en: {test_dir}")

    num_classes = len(labels)
    import numpy as np

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    total = len(samples)
    for index, (image_path, true_idx) in enumerate(samples, start=1):
        prediction = engine.predict_image(image_path)
        pred_idx = labels.index(prediction.class_name)
        cm[true_idx, pred_idx] += 1

        if on_progress is not None:
            on_progress(index, total)

    all_samples = int(cm.sum())
    correct = int(np.trace(cm))
    overall_accuracy = correct / all_samples if all_samples else 0.0

    per_class_accuracy: dict[str, float] = {}
    per_class_recall: dict[str, float] = {}

    for class_idx, label in enumerate(labels):
        tp = int(cm[class_idx, class_idx])
        row_total = int(cm[class_idx, :].sum())
        recall = tp / row_total if row_total > 0 else 0.0
        per_class_accuracy[label] = recall
        per_class_recall[label] = recall

    matrix_list = cm.tolist()
    cm_path = (
        output_image_path.resolve()
        if output_image_path
        else (project_root() / "reports" / "generated" / "confusion_matrix.png").resolve()
    )
    save_confusion_matrix(matrix_list, labels, cm_path)

    return EvalResults(
        accuracy=float(overall_accuracy),
        confusion_matrix=matrix_list,
        per_class_accuracy=per_class_accuracy,
        per_class_recall=per_class_recall,
        confusion_matrix_path=cm_path,
    )
