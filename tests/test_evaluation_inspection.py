from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from intelligest.cli import build_parser
from intelligest.evaluation import normalize_name, validate_test_structure


class EvaluationAndInspectionTests(unittest.TestCase):
    def test_normalize_name(self) -> None:
        self.assertEqual(normalize_name("Left-Arm Side"), "left_arm_side")
        self.assertEqual(normalize_name("  right_arm_up  "), "right_arm_up")

    def test_validate_test_structure_valid(self) -> None:
        labels = ("left_arm_side", "left_arm_up", "right_arm_side", "right_arm_up")
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_path = Path(tmp_dir)
            for label in labels:
                (test_path / label).mkdir()

            mapping = validate_test_structure(test_path, labels)
            self.assertEqual(len(mapping), 4)
            self.assertEqual(mapping[0].name, "left_arm_side")

    def test_validate_test_structure_invalid_count(self) -> None:
        labels = ("left_arm_side", "left_arm_up", "right_arm_side", "right_arm_up")
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_path = Path(tmp_dir)
            (test_path / "left_arm_side").mkdir()

            with self.assertRaises(ValueError):
                validate_test_structure(test_path, labels)

    def test_cli_parser_includes_subcommands(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["inspect-onnx", "--model", "models/custom.onnx", "--expected-classes", "7"])
        self.assertEqual(args.command, "inspect-onnx")
        self.assertEqual(args.expected_classes, 7)
        self.assertEqual(args.model, Path("models/custom.onnx"))

        args_eval = parser.parse_args(["evaluate-onnx", "--profile", "arm_poses_7"])
        self.assertEqual(args_eval.command, "evaluate-onnx")
        self.assertEqual(args_eval.profile, "arm_poses_7")


if __name__ == "__main__":
    unittest.main()
