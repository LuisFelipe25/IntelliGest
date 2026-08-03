from __future__ import annotations

import unittest
from pathlib import Path

from intelligest.config import (
    DatasetProfile,
    ModelContract,
    ToolchainConfig,
    project_root,
)


class ConfigurationTests(unittest.TestCase):
    def test_all_profiles_load_with_unique_classes(self) -> None:
        expected = {
            "arm_poses_7": 7,
        }
        for profile_id, class_count in expected.items():
            with self.subTest(profile=profile_id):
                profile = DatasetProfile.load(profile_id)
                self.assertEqual(len(profile.classes), class_count)
                self.assertEqual(len(set(profile.classes)), class_count)

    def test_model_contracts_match_profile_class_order(self) -> None:
        root = project_root()
        for path in sorted((root / "configs" / "models").glob("*.json")):
            with self.subTest(contract=path.name):
                contract = ModelContract.load(path)
                self.assertEqual(contract.classes, DatasetProfile.load(contract.profile).classes)

    def test_require_dataset_override(self) -> None:
        profile = DatasetProfile.load("arm_poses_7")
        override = Path("external-dataset")
        self.assertEqual(profile.require_dataset(override), override.resolve())
        self.assertTrue(profile.require_dataset().name, "arm_poses_cls")

    def test_toolchain_points_to_one_yolov5_copy(self) -> None:
        config = ToolchainConfig.load()
        self.assertEqual(config.yolov5_path, project_root() / "third_party" / "yolov5")
        self.assertTrue((config.yolov5_path / "classify" / "train.py").is_file())


if __name__ == "__main__":
    unittest.main()
