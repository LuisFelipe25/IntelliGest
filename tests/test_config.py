from __future__ import annotations

import unittest
from pathlib import Path

from intelligest.config import (
    ConfigurationError,
    DatasetProfile,
    ModelContract,
    ToolchainConfig,
    project_root,
)


class ConfigurationTests(unittest.TestCase):
    def test_all_profiles_load_with_unique_classes(self) -> None:
        expected = {"ciima_4": 4, "intelligest_8": 8, "yarvis_4": 4, "visio_8_legacy": 8}
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

    def test_missing_visio_dataset_requires_override(self) -> None:
        profile = DatasetProfile.load("visio_8_legacy")
        with self.assertRaises(ConfigurationError):
            profile.require_dataset()
        override = Path("external-dataset")
        self.assertEqual(profile.require_dataset(override), override.resolve())

    def test_toolchain_points_to_one_yolov5_copy(self) -> None:
        config = ToolchainConfig.load()
        self.assertEqual(config.yolov5_path, project_root() / "third_party" / "yolov5")
        self.assertTrue((config.yolov5_path / "classify" / "train.py").is_file())


if __name__ == "__main__":
    unittest.main()
