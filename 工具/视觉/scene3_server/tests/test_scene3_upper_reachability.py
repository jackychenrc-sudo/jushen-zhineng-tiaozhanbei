import importlib.util
import math
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scene3_upper_reachability.py"
SPEC = importlib.util.spec_from_file_location("scene3_upper_reachability", SCRIPT)
reachability = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reachability)


class UpperReachabilityTests(unittest.TestCase):
    def test_distance_grid_includes_minimum(self):
        values = reachability.distance_grid(1.437, 0.78, 0.05)
        self.assertAlmostEqual(1.437, values[0])
        self.assertAlmostEqual(0.78, values[-1])
        self.assertTrue(all(a > b for a, b in zip(values, values[1:])))

    def test_target_poses_apply_virtual_forward_motion(self):
        virtual, pregrasp, grasp = reachability.target_poses(
            [1.437, -0.25, 0.24],
            current_shelf_distance=1.437,
            candidate_shelf_distance=0.887,
            hand_reference_length=0.193,
            surface_clearance=0.015,
            pregrasp_clearance=0.12,
            y_offset=0.0,
            z_offset=0.01,
        )
        self.assertEqual([0.887, -0.25, 0.25], [round(v, 3) for v in virtual])
        self.assertEqual([0.559, -0.25, 0.25], [round(v, 3) for v in pregrasp])
        self.assertEqual([0.679, -0.25, 0.25], [round(v, 3) for v in grasp])

    def test_quaternion_is_normalized(self):
        quaternion = reachability.quaternion_from_rpy(0.0, -math.pi / 2.0, 0.0)
        norm = math.sqrt(sum(value * value for value in quaternion))
        self.assertAlmostEqual(1.0, norm)
        self.assertAlmostEqual(0.0, quaternion[0])
        self.assertAlmostEqual(-math.sqrt(0.5), quaternion[1])

    def test_choose_target_uses_right_hand_lateral_position(self):
        trays = [
            {"id": "upper_x0", "base_link_xyz_raw_m": [1.4, 0.4, 0.2]},
            {"id": "upper_x1", "base_link_xyz_raw_m": [1.4, -0.25, 0.2]},
            {"id": "upper_x2", "base_link_xyz_raw_m": [1.4, -0.45, 0.2]},
        ]
        target, preferred = reachability.choose_target(trays, "right", 0.28, -0.28)
        self.assertEqual("upper_x1", target["id"])
        self.assertAlmostEqual(-0.28, preferred)


if __name__ == "__main__":
    unittest.main()

