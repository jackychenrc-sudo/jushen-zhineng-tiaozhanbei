#!/usr/bin/env python3

import os
import sys
import unittest

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
SERVER_DIR = os.path.dirname(HERE)
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)

import scene3_wrist_depth_step as depth_step


class WristDepthStepGeometryTest(unittest.TestCase):
    def test_initial_prompt_targets_robot_facing_edge(self):
        head = np.array([0.6435, -0.2334, 0.2517])
        prompt, direction = depth_step.initial_edge_prompt(head, 0.10)

        self.assertAlmostEqual(0.10, np.linalg.norm(head[:2] - prompt[:2]))
        self.assertAlmostEqual(head[2], prompt[2])
        self.assertGreater(np.dot(head[:2] - prompt[:2], direction), 0.099)
        np.testing.assert_allclose(prompt, [0.5495, -0.1993, 0.2517], atol=0.001)

    def test_initial_candidate_accepts_observed_near_edge(self):
        head = np.array([0.6485, -0.2349, 0.2536])
        prompt, direction = depth_step.initial_edge_prompt(head, 0.10)
        checks, details = depth_step.initial_candidate_gate(
            [0.5524, -0.2084, 0.2490],
            head,
            prompt,
            direction,
        )

        self.assertTrue(all(checks.values()), (checks, details))
        self.assertAlmostEqual(0.10, details["edge_offset_m"], delta=0.015)

    def test_initial_candidate_rejects_far_tray_body_or_rack(self):
        head = np.array([0.6485, -0.2349, 0.2536])
        prompt, direction = depth_step.initial_edge_prompt(head, 0.10)
        checks, _ = depth_step.initial_candidate_gate(
            [0.645, -0.235, 0.34],
            head,
            prompt,
            direction,
        )

        self.assertFalse(all(checks.values()))
        self.assertFalse(checks["near_edge_prompt"])
        self.assertFalse(checks["robot_facing_edge"])
        self.assertFalse(checks["height_consistency"])


if __name__ == "__main__":
    unittest.main()

