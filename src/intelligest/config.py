from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ConfigurationError(ValueError):
    """Raised when a project configuration is incomplete or inconsistent."""


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigurationError(f"No existe la configuración: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigurationError(f"JSON inválido en {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigurationError(f"La raíz de {path} debe ser un objeto JSON")
    return value


def resolve_path(value: str | None, root: Path | None = None) -> Path | None:
    if value is None:
        return None
    root = root or project_root()
    expanded = Path(os.path.expandvars(os.path.expanduser(value)))
    return expanded.resolve() if expanded.is_absolute() else (root / expanded).resolve()


def local_paths(root: Path | None = None) -> dict[str, Any]:
    root = root or project_root()
    path = root / "configs" / "paths.local.json"
    return load_json(path) if path.is_file() else {}


@dataclass(frozen=True)
class DatasetProfile:
    id: str
    classes: tuple[str, ...]
    status: str
    dataset_path: Path | None
    seed: int

    @classmethod
    def load(cls, profile: str, root: Path | None = None) -> DatasetProfile:
        root = root or project_root()
        data = load_json(root / "configs" / "datasets" / f"{profile}.json")
        classes = tuple(str(item) for item in data.get("classes", []))
        if not classes or len(classes) != len(set(classes)):
            raise ConfigurationError(f"El perfil {profile} necesita clases únicas y ordenadas")
        overrides = local_paths(root).get("datasets", {})
        configured_path = overrides.get(profile, data.get("dataset_path"))
        return cls(
            id=str(data["id"]),
            classes=classes,
            status=str(data.get("status", "unknown")),
            dataset_path=resolve_path(configured_path, root),
            seed=int(data.get("training_defaults", {}).get("seed", 0)),
        )

    def require_dataset(self, override: Path | None = None) -> Path:
        path = override.resolve() if override else self.dataset_path
        if path is None:
            raise ConfigurationError(f"El perfil {self.id} requiere una ruta explícita con --dataset")
        return path


@dataclass(frozen=True)
class ModelContract:
    id: str
    profile: str
    path: Path | None
    classes: tuple[str, ...]
    shape: tuple[int, ...]
    layout: str
    scale: float
    mean: tuple[float, ...]
    std: tuple[float, ...]
    color: str

    @classmethod
    def load(cls, contract_path: Path, root: Path | None = None) -> ModelContract:
        root = root or project_root()
        data = load_json(contract_path)
        profile_id = str(data["profile"])
        overrides = local_paths(root).get("models", {})
        configured_path = overrides.get(profile_id, data.get("path"))
        contract = cls(
            id=str(data["id"]),
            profile=profile_id,
            path=resolve_path(configured_path, root),
            classes=tuple(str(item) for item in data["classes"]),
            shape=tuple(int(item) for item in data["input"]["shape"]),
            layout=str(data["input"]["layout"]),
            scale=float(data["normalization"]["scale"]),
            mean=tuple(float(item) for item in data["normalization"]["mean"]),
            std=tuple(float(item) for item in data["normalization"]["std"]),
            color=str(data["normalization"]["color"]),
        )
        profile = DatasetProfile.load(contract.profile, root)
        if contract.classes != profile.classes:
            raise ConfigurationError(
                f"El orden de clases del modelo {contract.id} no coincide con {profile.id}"
            )
        if contract.layout not in {"NCHW", "NHWC"} or len(contract.shape) != 4:
            raise ConfigurationError(f"Contrato de entrada inválido para {contract.id}")
        if contract.color != "RGB" or contract.scale <= 0:
            raise ConfigurationError(f"Normalización no soportada para {contract.id}")
        return contract

    def require_model(self, override: Path | None = None) -> Path:
        path = override.resolve() if override else self.path
        if path is None:
            raise ConfigurationError(f"El contrato {self.id} requiere una ruta de modelo")
        if not path.is_file():
            raise ConfigurationError(f"No existe el modelo configurado: {path}")
        return path


@dataclass(frozen=True)
class ToolchainConfig:
    yolov5_path: Path
    source_commit: str | None
    upstream_base_commit: str | None

    @classmethod
    def load(cls, root: Path | None = None) -> ToolchainConfig:
        root = root or project_root()
        data = load_json(root / "configs" / "toolchain.json")
        override = local_paths(root).get("yolov5_path", data.get("yolov5_path"))
        path = resolve_path(override, root)
        if path is None:
            raise ConfigurationError("Falta yolov5_path en la configuración")
        return cls(path, data.get("source_commit"), data.get("upstream_base_commit"))
