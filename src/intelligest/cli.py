from __future__ import annotations

import argparse
import json
from pathlib import Path

from intelligest.config import DatasetProfile, ToolchainConfig, project_root
from intelligest.export.onnx import build_export_command
from intelligest.training.yolov5 import (
    build_evaluate_command,
    build_train_command,
    run,
    write_experiment_record,
)


def _json(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    root = project_root()
    parser = argparse.ArgumentParser(prog="intelligest", description="IntelliGest consolidated CLI")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("profiles", help="List profiles, classes and configured external datasets")
    commands.add_parser("paths", help="Show the external YOLOv5, dataset and model locations")

    train = commands.add_parser("train", help="Run or print the external YOLOv5 training command")
    train.add_argument("--profile", required=True)
    train.add_argument("--dataset", type=Path, help="Override the dataset path configured by the profile")
    train.add_argument("--model", default="yolov5n-cls.pt")
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--batch-size", type=int, default=8)
    train.add_argument("--imgsz", type=int, default=224)
    train.add_argument("--device", default="cpu")
    train.add_argument("--seed", type=int)
    train.add_argument("--output", type=Path, default=root / "runs")
    train.add_argument("--execute", action="store_true", help="Execute; otherwise only print the command")

    evaluate = commands.add_parser("evaluate", help="Run or print classification evaluation")
    evaluate.add_argument("--weights", type=Path, required=True)
    evaluate.add_argument("--dataset", type=Path, required=True)
    evaluate.add_argument("--imgsz", type=int, default=224)
    evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument("--execute", action="store_true")

    export = commands.add_parser("export-onnx", help="Run or print ONNX export")
    export.add_argument("--weights", type=Path, required=True)
    export.add_argument("--imgsz", type=int, default=224)
    export.add_argument("--execute", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = project_root()
    if args.command == "profiles":
        values = []
        for path in sorted((root / "configs" / "datasets").glob("*.json")):
            profile = DatasetProfile.load(path.stem)
            values.append(
                {
                    "id": profile.id,
                    "status": profile.status,
                    "classes": list(profile.classes),
                    "dataset": str(profile.dataset_path) if profile.dataset_path else None,
                }
            )
        _json(values)
        return 0
    if args.command == "paths":
        toolchain = ToolchainConfig.load(root)
        config_paths = sorted((root / "configs" / "datasets").glob("*.json"))
        profiles = [DatasetProfile.load(path.stem) for path in config_paths]
        _json(
            {
                "yolov5": str(toolchain.yolov5_path),
                "profiles": {
                    profile.id: str(profile.dataset_path) if profile.dataset_path else None
                    for profile in profiles
                },
            }
        )
        return 0
    if args.command == "train":
        profile = DatasetProfile.load(args.profile)
        dataset = profile.require_dataset(args.dataset)
        if args.execute and not dataset.is_dir():
            raise FileNotFoundError(f"No existe el dataset: {dataset}")
        seed = profile.seed if args.seed is None else args.seed
        command = build_train_command(
            profile,
            dataset,
            args.model,
            args.epochs,
            args.batch_size,
            args.imgsz,
            args.device,
            seed,
            args.output,
        )
        if args.execute:
            write_experiment_record(args.output / profile.id / "experiment.json", profile, dataset, command)
        return run(command, args.execute)
    if args.command == "evaluate":
        if args.execute and not args.weights.is_file():
            raise FileNotFoundError(f"No existen los pesos: {args.weights}")
        if args.execute and not args.dataset.is_dir():
            raise FileNotFoundError(f"No existe el dataset: {args.dataset}")
        return run(build_evaluate_command(args.weights, args.dataset, args.imgsz, args.device), args.execute)
    if args.command == "export-onnx":
        if args.execute and not args.weights.is_file():
            raise FileNotFoundError(f"No existen los pesos: {args.weights}")
        return run(build_export_command(args.weights, args.imgsz), args.execute)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
