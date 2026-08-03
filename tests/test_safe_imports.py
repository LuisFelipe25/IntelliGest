from __future__ import annotations

import importlib
import unittest


class SafeImportTests(unittest.TestCase):
    def test_optional_runtime_modules_are_lazy(self) -> None:
        for module in (
            "intelligest.app.desktop",
            "intelligest.inference.engine",
            "intelligest.training.yolov5",
            "intelligest.export.onnx",
            "intelligest.evaluation",
            "intelligest.inspection",
        ):
            with self.subTest(module=module):
                self.assertIsNotNone(importlib.import_module(module))


if __name__ == "__main__":
    unittest.main()
