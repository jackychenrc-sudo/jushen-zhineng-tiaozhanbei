import sys
import unittest
from pathlib import Path

import numpy as np


SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

import scene3_fixed_wrist_ik as fixed_ik


class FixedWristIkTests(unittest.TestCase):
    @staticmethod
    def synthetic_fk(arm):
        q = np.asarray(arm, dtype=float)[7:11]
        return np.array(
            [
                0.46 + 0.10 * np.sin(q[0]) + 0.04 * np.sin(q[2]),
                -0.22 + 0.08 * np.sin(q[1]) + 0.03 * np.sin(q[3]),
                0.31 + 0.07 * np.cos(q[0]) - 0.06 * np.sin(q[2]) + 0.05 * np.cos(q[3]),
            ]
        )

    def test_converges_without_changing_wrist_or_left_arm(self):
        current = np.deg2rad(
            [0, 0, 0, 0, 0, 0, 0, -31, 20, -23, -60, 15, -7, -22]
        )
        goal = current.copy()
        goal[7:11] += np.deg2rad([8, -6, 7, -9])
        target = self.synthetic_fk(goal)

        result = fixed_ik.solve_fixed_wrist_position(
            self.synthetic_fk,
            current,
            target,
            tolerance_m=0.001,
            max_iterations=40,
        )

        self.assertTrue(result["success"])
        self.assertLessEqual(result["final_error_m"], 0.001)
        np.testing.assert_array_equal(result["arm_joints_rad"][:7], current[:7])
        np.testing.assert_array_equal(result["arm_joints_rad"][11:14], current[11:14])
        np.testing.assert_array_equal(result["wrist_delta_rad"], np.zeros(3))

    def test_rejects_wrong_arm_vector_size(self):
        with self.assertRaises(ValueError):
            fixed_ik.solve_fixed_wrist_position(
                self.synthetic_fk,
                np.zeros(13),
                np.zeros(3),
            )


if __name__ == "__main__":
    unittest.main()

