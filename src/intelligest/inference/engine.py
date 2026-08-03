from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from intelligest.config import ModelContract


@dataclass(frozen=True)
class Prediction:
    class_name: str
    confidence: float
    probabilities: tuple[float, ...]
    infer_ms: float = 0.0


class ONNXEngine:
    def __init__(
        self,
        contract: ModelContract,
        provider: str = "CPU",
        model_path: Path | None = None,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("Instala IntelliGest con el extra inference para usar ONNX Runtime") from exc

        resolved_model = contract.require_model(model_path)
        providers = {
            "CPU": ["CPUExecutionProvider"],
            "CUDA": ["CUDAExecutionProvider", "CPUExecutionProvider"],
            "DirectML": ["DmlExecutionProvider", "CPUExecutionProvider"],
        }.get(provider)
        if providers is None:
            raise ValueError("provider debe ser CPU, CUDA o DirectML")
        self.contract = contract
        self.session = ort.InferenceSession(str(resolved_model), providers=providers)
        self.input = self.session.get_inputs()[0]
        self.output = self.session.get_outputs()[0]
        output_shape = list(self.output.shape)
        if (
            output_shape
            and output_shape[-1] is not None
            and isinstance(output_shape[-1], int)
            and output_shape[-1] > 0
        ):
            output_classes = output_shape[-1]
        else:
            import numpy as np

            h, w = self._image_size()
            dummy = (
                np.zeros((1, 3, h, w), dtype=np.float32)
                if self.contract.layout == "NCHW"
                else np.zeros((1, h, w, 3), dtype=np.float32)
            )
            raw = self.session.run([self.output.name], {self.input.name: dummy})[0]
            output_classes = raw.reshape(1, -1).shape[-1]

        if output_classes != len(contract.classes):
            raise ValueError(
                f"Salida ONNX ({output_classes} clases) incompatible con {len(contract.classes)} clases del contrato"
            )

    def preprocess_bgr(self, frame):
        try:
            import cv2
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("OpenCV y NumPy son necesarios para preprocesar imágenes") from exc
        height, width = self._image_size()
        resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        value = rgb.astype(np.float32) * self.contract.scale
        mean = np.asarray(self.contract.mean, dtype=np.float32)
        std = np.asarray(self.contract.std, dtype=np.float32)
        value = (value - mean) / std
        if self.contract.layout == "NCHW":
            value = np.transpose(value, (2, 0, 1))
        return np.expand_dims(value, axis=0).astype(np.float32)

    def _image_size(self) -> tuple[int, int]:
        if self.contract.layout == "NCHW":
            return self.contract.shape[2], self.contract.shape[3]
        return self.contract.shape[1], self.contract.shape[2]

    def predict_bgr(self, frame) -> Prediction:
        import numpy as np

        tensor = self.preprocess_bgr(frame)
        start = time.perf_counter()
        raw_output = self.session.run([self.output.name], {self.input.name: tensor})[0]
        infer_ms = (time.perf_counter() - start) * 1000.0

        values = raw_output.reshape(1, -1)
        shifted = values - np.max(values, axis=1, keepdims=True)
        probabilities = (np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True))[0]
        index = int(np.argmax(probabilities))
        return Prediction(
            class_name=self.contract.classes[index],
            confidence=float(probabilities[index]),
            probabilities=tuple(float(item) for item in probabilities),
            infer_ms=infer_ms,
        )

    def predict_image(self, path: Path) -> Prediction:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("OpenCV es necesario para cargar imágenes") from exc
        frame = cv2.imread(str(path))
        if frame is None:
            raise ValueError(f"No se pudo leer la imagen: {path}")
        return self.predict_bgr(frame)

    def smoke_test(self) -> dict[str, object]:
        import numpy as np

        height, width = self._image_size()
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        prediction = self.predict_bgr(frame)
        return {
            "model": self.contract.id,
            "input_name": self.input.name,
            "input_shape": list(self.input.shape),
            "output_name": self.output.name,
            "output_shape": list(self.output.shape),
            "class": prediction.class_name,
            "confidence": prediction.confidence,
            "infer_ms": prediction.infer_ms,
            "probability_sum": sum(prediction.probabilities),
        }
