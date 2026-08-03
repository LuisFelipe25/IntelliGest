from __future__ import annotations

import unittest

from intelligest.config import project_root
from intelligest.integrations.udp import UDPActionConfig


class UDPConfigurationTests(unittest.TestCase):
    def test_arm_poses_7_mapping_without_network(self) -> None:
        config = UDPActionConfig.load(project_root() / "configs" / "actions" / "arm_poses_7.json")
        self.assertEqual(config.payload_for("left_arm_side"), b"left_side")
        self.assertEqual(config.payload_for("right_arm_up"), b"right_up")
        with self.assertRaises(ValueError):
            config.payload_for("unknown")


if __name__ == "__main__":
    unittest.main()
