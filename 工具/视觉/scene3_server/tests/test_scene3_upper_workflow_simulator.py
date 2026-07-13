import sys
import unittest
from argparse import Namespace
from pathlib import Path


SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

import scene3_upper_workflow_simulator as simulator


def settings(scenario, **overrides):
    values = {
        "scenario": scenario,
        "seed": 5,
        "initial_distance": 1.30,
        "desired_distance": 0.90,
        "maximum_cycles": 12,
        "motion_gain": 0.72,
        "motion_noise": 0.003,
        "output": "unused.json",
    }
    values.update(overrides)
    return Namespace(**values)


class UpperWorkflowSimulatorTests(unittest.TestCase):
    def test_nominal_approach_stops_for_real_calibration(self):
        payload = simulator.run_simulation(settings("nominal"))

        self.assertEqual("calibration_required", payload["terminal_status"])
        self.assertGreater(payload["simulated_forward_total_m"], 0.0)
        self.assertFalse(payload["robot_commands_sent"])
        self.assertFalse(payload["physics_validated"])
        pulses = [
            cycle["motion_abstraction"]["commanded_forward_m"]
            for cycle in payload["cycles"]
            if "motion_abstraction" in cycle
        ]
        self.assertTrue(pulses)
        self.assertLessEqual(max(pulses), 0.08)
        self.assertEqual(
            "stop_before_arm_or_body_command", payload["cycles"][-1]["decision"]
        )

    def test_full_dry_run_reaches_verification_states_without_commands(self):
        payload = simulator.run_simulation(settings("full_dry_run"))

        self.assertEqual("ready_dry_run", payload["terminal_status"])
        self.assertTrue(payload["model"]["synthetic_calibration_used"])
        self.assertFalse(payload["robot_commands_sent"])
        reached = {
            item["name"]
            for item in payload["cycles"][-1]["simulated_state_results"]
        }
        self.assertIn("VERIFY_REMOVAL", reached)
        self.assertIn("VERIFY_OUTBOUND", reached)

    def test_unstable_vision_blocks_motion(self):
        payload = simulator.run_simulation(settings("unstable_vision"))

        self.assertEqual("vision_safety_stop", payload["terminal_status"])
        self.assertEqual(0.0, payload["simulated_forward_total_m"])
        self.assertEqual("stop_before_motion", payload["cycles"][0]["decision"])

    def test_missed_detection_blocks_motion(self):
        payload = simulator.run_simulation(settings("missed_detection"))

        self.assertEqual("vision_safety_stop", payload["terminal_status"])
        self.assertEqual(0.0, payload["simulated_forward_total_m"])
        self.assertEqual("stop_before_motion", payload["cycles"][0]["decision"])

    def test_too_close_requests_reposition_without_forward_motion(self):
        payload = simulator.run_simulation(settings("too_close"))

        self.assertEqual("reposition_required", payload["terminal_status"])
        self.assertEqual(0.0, payload["simulated_forward_total_m"])
        self.assertEqual(
            "stop_before_arm_or_body_command", payload["cycles"][0]["decision"]
        )


if __name__ == "__main__":
    unittest.main()
