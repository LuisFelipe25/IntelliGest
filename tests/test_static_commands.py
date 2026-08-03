from __future__ import annotations

import unittest
from pathlib import Path

from intelligest.config import DatasetProfile
from intelligest.export.onnx import build_export_command
from intelligest.training.yolov5 import build_train_command


class StaticCommandTests(unittest.TestCase):
    def test_train_command_uses_single_external_checkout(self) -> None:
        profile = DatasetProfile.load("arm_poses_7")
        command = build_train_command(
            profile,
            profile.require_dataset(),
            "yolov5n-cls.pt",
            1,
            2,
            224,
            "cpu",
            0,
            profile.require_dataset().parent / "runs-static-test",
        )
        self.assertIn("classify", command[1])
        self.assertIn("train.py", command[1])
        self.assertIn("--data", command)

    def test_export_command_targets_external_export_module(self) -> None:
        weights = Path("external-model.pt")
        command = build_export_command(weights)
        self.assertTrue(command[1].endswith("export.py"))
        self.assertEqual(command[command.index("--weights") + 1], str(weights))


if __name__ == "__main__":
    unittest.main()
