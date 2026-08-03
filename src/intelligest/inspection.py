from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ONNXInspection:
    model_path: Path
    input_name: str
    input_shape: list[object]
    input_type: str
    output_name: str
    output_shape: list[object]
    output_type: str
    detected_classes: int

    def to_dict(self) -> dict[str, object]:
        return {
            "model_path": str(self.model_path),
            "input_name": self.input_name,
            "input_shape": self.input_shape,
            "input_type": self.input_type,
            "output_name": self.output_name,
            "output_shape": self.output_shape,
            "output_type": self.output_type,
            "detected_classes": self.detected_classes,
        }


def inspect_onnx_model(
    model_path: Path, expected_classes: int | None = None
) -> ONNXInspection:
    resolved = model_path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"No existe el archivo de modelo ONNX: {model_path}")

    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("ONNX Runtime es necesario para inspeccionar el modelo") from exc

    session = ort.InferenceSession(str(resolved), providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    outputs = session.get_outputs()

    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError(
            f"El modelo ONNX debe tener exactamente 1 entrada y 1 salida. Encontradas: {len(inputs)} in, {len(outputs)} out."
        )

    inp = inputs[0]
    out = outputs[0]

    output_shape = list(out.shape)
    detected_classes = output_shape[-1] if output_shape else None

    if not isinstance(detected_classes, int) or detected_classes <= 0:
        import numpy as np

        dummy = np.zeros((1, 3, 224, 224), dtype=np.float32)
        try:
            raw = session.run([out.name], {inp.name: dummy})[0]
            detected_classes = int(raw.reshape(1, -1).shape[-1])
        except Exception as exc:
            raise ValueError(f"No se pudo determinar el número de clases de salida: {exc}") from exc

    inspection = ONNXInspection(
        model_path=resolved,
        input_name=inp.name,
        input_shape=list(inp.shape),
        input_type=inp.type,
        output_name=out.name,
        output_shape=list(out.shape),
        output_type=out.type,
        detected_classes=detected_classes,
    )

    if expected_classes is not None and detected_classes != expected_classes:
        raise ValueError(
            f"Clases detectadas en el modelo ({detected_classes}) no coinciden con las esperadas ({expected_classes})"
        )

    return inspection
