"""Verify that legacy datasets, models, and YOLOv5 were preserved byte-for-byte."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
SOURCE_ROOT = WORKSPACE_ROOT / "work" / "repos"
MODEL_SUFFIXES = {".onnx", ".pt", ".pth"}
REPOSITORIES = (
    "IntelliGest",
    "CIIMA_Visio_AI",
    "Visio_AI",
    "training_model",
    "YARVIS",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory(root: Path, files: list[Path]) -> dict[str, dict[str, object]]:
    with ThreadPoolExecutor(max_workers=16) as executor:
        hashes = executor.map(sha256, files)
        return {
            path.relative_to(root).as_posix(): {
                "size": path.stat().st_size,
                "sha256": digest,
            }
            for path, digest in zip(files, hashes, strict=True)
        }


def compare(source_root: Path, destination_root: Path, source_files: list[Path]) -> dict[str, object]:
    destination_files = sorted(path for path in destination_root.rglob("*") if path.is_file())
    source = inventory(source_root, source_files)
    destination = inventory(destination_root, destination_files)
    source_names = set(source)
    destination_names = set(destination)
    common = source_names & destination_names
    mismatched = sorted(name for name in common if source[name] != destination[name])
    missing = sorted(source_names - destination_names)
    extra = sorted(destination_names - source_names)
    return {
        "source_files": len(source),
        "destination_files": len(destination),
        "source_bytes": sum(int(item["size"]) for item in source.values()),
        "destination_bytes": sum(int(item["size"]) for item in destination.values()),
        "missing": missing,
        "extra": extra,
        "mismatched": mismatched,
        "verified": not (missing or extra or mismatched),
    }


def yolov5_source_files(root: Path) -> list[Path]:
    excluded_dirs = {".git", "datasets", "runs"}
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and not any(part in excluded_dirs for part in path.relative_to(root).parts)
        and path.suffix.lower() not in MODEL_SUFFIXES
    )


def main() -> int:
    report: dict[str, object] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "algorithm": "SHA-256",
        "repositories": {},
    }
    repositories = report["repositories"]
    assert isinstance(repositories, dict)

    for name in REPOSITORIES:
        source_repo = SOURCE_ROOT / name
        repo_result: dict[str, object] = {}
        source_dataset = source_repo / "datasets"
        if source_dataset.is_dir():
            files = sorted(path for path in source_dataset.rglob("*") if path.is_file())
            repo_result["datasets"] = compare(
                source_dataset,
                PROJECT_ROOT / "data" / "legacy" / name / "datasets",
                files,
            )

        model_files = sorted(
            path
            for path in source_repo.rglob("*")
            if path.is_file() and path.suffix.lower() in MODEL_SUFFIXES
        )
        if model_files:
            repo_result["models"] = compare(
                source_repo,
                PROJECT_ROOT / "models" / "legacy" / name,
                model_files,
            )
        repositories[name] = repo_result

    yolov5_source = SOURCE_ROOT / "YARVIS"
    report["yolov5"] = compare(
        yolov5_source,
        PROJECT_ROOT / "third_party" / "yolov5",
        yolov5_source_files(yolov5_source),
    )

    sections = [
        section
        for repository in repositories.values()
        for section in repository.values()
    ]
    sections.append(report["yolov5"])
    report["verified"] = all(section["verified"] for section in sections)

    output = PROJECT_ROOT / "reports" / "preservation-verification.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
