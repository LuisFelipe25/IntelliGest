from __future__ import annotations

import unittest

from intelligest.config import project_root
from intelligest.integrations.udp import UDPActionConfig


class UDPConfigurationTests(unittest.TestCase):
    def test_ciima_mapping_without_network(self) -> None:
        config = UDPActionConfig.load(project_root() / "configs" / "actions" / "ciima_4.json")
        self.assertEqual(config.payload_for("left_arm_side"), b"a")
        self.assertEqual(config.payload_for("right_arm_up"), b"d")
        with self.assertRaises(ValueError):
            config.payload_for("unknown")


if __name__ == "__main__":
    unittest.main()
