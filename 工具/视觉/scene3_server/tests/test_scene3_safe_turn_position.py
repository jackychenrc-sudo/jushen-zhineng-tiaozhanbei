#!/usr/bin/env python3

import os
import sys
import unittest

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
SERVER = os.path.dirname(HERE)
if SERVER not in sys.path:
    sys.path.insert(0, SERVER)

from scene3_safe_turn_position import (
    candidate_steps,
    cartesian_metrics,
    predict_physical_tcp,
    segment_checks,
    tcp_step_target,
)


class SafeTurnPositionTest(unittest.TestCase):

    def test_tcp_step_moves_eef_towards_physical_target(self):
        result = tcp_step_target(
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.4, 0.0, 0.0],
            0.02,
        )
        np.testing.assert_allclose(result["eef_target"], [0.02, 0.0, 0.0])
        np.testing.assert_allclose(result["direction"], [1.0, 0.0, 0.0])
        self.assertAlmostEqual(result["step"], 0.02)
        self.assertAlmostEqual(result["remaining"], 0.3)

    def test_predicted_tcp_preserves_rigid_eef_offset(self):
        predicted = predict_physical_tcp(
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.1, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        )
        np.testing.assert_allclose(predicted, [0.3, 0.0, 0.0])

    def test_cartesian_gate_accepts_clean_two_centimetre_step(self):
        metrics = cartesian_metrics(
            [0.0, 0.0, 0.0],
            [0.019, 0.001, 0.0],
            [0.20, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        )
        checks = segment_checks(0.02, 0.20, metrics)
        self.assertTrue(all(checks.values()))

    def test_candidate_steps_halves_until_minimum(self):
        self.assertEqual(candidate_steps(0.02, 0.005, 0.2), [0.02, 0.01, 0.005])


if __name__ == "__main__":
    unittest.main()

