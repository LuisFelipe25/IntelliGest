from __future__ import annotations

import argparse
import json
from pathlib import Path

from intelligest.config import DatasetProfile, ModelContract, ToolchainConfig, project_root
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

    inspect = commands.add_parser("inspect-onnx", help="Inspect ONNX model input/output metadata and class count")
    inspect.add_argument("--model", type=Path, required=True, help="ONNX model path")
    inspect.add_argument("--expected-classes", type=int, help="Expected number of output classes")

    eval_onnx = commands.add_parser("evaluate-onnx", help="Run offline ONNX dataset evaluation and plot confusion matrix")
    eval_onnx.add_argument("--model", type=Path, help="Override model file path")
    eval_onnx.add_argument("--contract", type=Path, help="Model contract JSON path")
    eval_onnx.add_argument("--profile", default="arm_poses_7", help="Dataset profile (e.g. arm_poses_7)")
    eval_onnx.add_argument("--dataset", type=Path, help="Test dataset directory (defaults to test/ inside profile dataset path)")
    eval_onnx.add_argument("--eval-out", type=Path, help="Output file for confusion matrix PNG")
    eval_onnx.add_argument("--provider", choices=["CPU", "CUDA", "DirectML"], default="CPU")

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
    if args.command == "inspect-onnx":
        from intelligest.inspection import inspect_onnx_model

        result = inspect_onnx_model(args.model, args.expected_classes)
        _json(result.to_dict())
        return 0
    if args.command == "evaluate-onnx":
        from intelligest.evaluation import evaluate_dataset
        from intelligest.inference.engine import ONNXEngine

        contract_path = (
            args.contract
            if args.contract
            else (root / "configs" / "models" / f"{args.profile}_app.json")
        )
        contract = ModelContract.load(contract_path)
        engine = ONNXEngine(contract, provider=args.provider, model_path=args.model)

        if args.dataset:
            test_dir = args.dataset
        else:
            profile = DatasetProfile.load(contract.profile)
            ds_path = profile.require_dataset()
            test_dir = ds_path / "test" if (ds_path / "test").is_dir() else ds_path

        def progress(done: int, total: int) -> None:
            if done == total or done % 200 == 0:
                print(f"Evaluando: {done}/{total}")

        eval_res = evaluate_dataset(engine, test_dir, output_image_path=args.eval_out, on_progress=progress)
        _json(eval_res.to_dict())
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
