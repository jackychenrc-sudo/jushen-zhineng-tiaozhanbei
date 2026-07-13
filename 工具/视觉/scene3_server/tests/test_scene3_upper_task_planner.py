import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path


SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

import scene3_upper_approach as approach
import scene3_upper_task_planner as planner


def settings(**overrides):
    values = {
        "hand": "right",
        "preferred_left_y": 0.28,
        "preferred_right_y": -0.28,
        "desired_shelf_distance": 0.90,
        "approach_tolerance": 0.03,
        "maximum_forward_pulse": 0.08,
    }
    values.update(overrides)
    return Namespace(**values)


def stable_detection(distance):
    trays = []
    for index, lateral_y in enumerate((0.30, 0.0, -0.29)):
        trays.append(
            {
                "id": "upper_x{}".format(index),
                "geometry_source": "rgbd_vertical",
                "object_bbox": [500 + index * 100, 220, 518 + index * 100, 275],
                "center_pixel": [509 + index * 100, 247],
                "base_link_xyz_raw_m": [distance + index * 0.002, lateral_y, 0.27],
                "base_link_xyz_m": [distance + index * 0.002, lateral_y, 0.27],
                "selection_score": 0.80 + index * 0.01,
                "depth_shape_score": 0.90,
            }
        )
    return {
        "status": "ok",
        "algorithm": "scene3_temporal_consensus_v1",
        "source_frame": "Head Camera View",
        "target_frame": "base_link",
        "temporal_validation": {"passed": True, "errors": []},
        "upper_trays": trays,
    }


def calibrated_config():
    return {
        "calibrated": True,
        "hand": "right",
        "pregrasp_offset_xyz_m": [-0.10, 0.0, 0.02],
        "grasp_offset_xyz_m": [-0.02, 0.0, 0.01],
        "withdraw_offset_xyz_m": [-0.14, 0.0, 0.04],
        "end_effector_rpy_deg": [0.0, 90.0, 0.0],
        "outbound_prepose_xyz_m": [0.45, -0.30, 0.55],
        "outbound_release_xyz_m": [0.42, -0.30, 0.45],
        "outbound_rpy_deg": [0.0, 90.0, 0.0],
        "claw_open_positions": [0.0, 100.0],
        "claw_close_positions": [100.0, 100.0],
        "claw_velocity": [90.0, 90.0],
        "claw_effort": [1.0, 1.0],
    }


class UpperTaskPlannerTests(unittest.TestCase):
    def test_far_shelf_requires_one_bounded_pulse(self):
        data = stable_detection(1.30)

        payload = planner.build_plan(settings(), data, data["upper_trays"], {"calibrated": False})

        self.assertEqual("approach_required", payload["status"])
        self.assertEqual(0.08, payload["planned_forward_pulse_m"])
        self.assertFalse(payload["execution_enabled"])
        self.assertEqual("upper_x2", payload["selected_target"]["id"])
        self.assertEqual("REACQUIRE_TEMPORAL_VISION", payload["states"][-1]["name"])

    def test_too_close_shelf_is_blocked(self):
        data = stable_detection(0.82)

        payload = planner.build_plan(settings(), data, data["upper_trays"], calibrated_config())

        self.assertEqual("reposition_required", payload["status"])
        self.assertEqual(0.0, payload["planned_forward_pulse_m"])
        self.assertEqual("RETREAT_FROM_SHELF", payload["states"][-1]["name"])

    def test_near_shelf_requires_calibration(self):
        data = stable_detection(0.90)

        payload = planner.build_plan(settings(), data, data["upper_trays"], {"calibrated": False})

        self.assertEqual("calibration_required", payload["status"])
        self.assertEqual("ARM_CALIBRATION", payload["states"][-1]["name"])

    def test_calibrated_plan_computes_dry_run_poses(self):
        data = stable_detection(0.90)

        payload = planner.build_plan(settings(), data, data["upper_trays"], calibrated_config())

        self.assertEqual("ready_dry_run", payload["status"])
        self.assertFalse(payload["execution_enabled"])
        commands = {
            item["name"]: item.get("dry_run_command") for item in payload["states"]
        }
        for actual, expected in zip(
            commands["ARM_PREGRASP"]["position_xyz_m"], [0.804, -0.29, 0.29]
        ):
            self.assertAlmostEqual(expected, actual)
        self.assertEqual([90.0, 90.0], commands["CLAW_CLOSE"]["velocity"])
        self.assertEqual([1.0, 1.0], commands["CLAW_CLOSE"]["effort"])

    def test_unstable_input_is_rejected(self):
        data = stable_detection(0.90)
        data["temporal_validation"]["passed"] = False

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "unstable.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not trusted"):
                planner.load_stable_detection(path)

    def test_approach_requires_temporal_consensus(self):
        single = stable_detection(1.30)
        single["algorithm"] = "multiscale_rgbd_object_geometry_v5"
        single.pop("temporal_validation")

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "single.json"
            path.write_text(json.dumps(single), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "requires scene3_temporal"):
                approach.load_detection(path)
            _, trays = approach.load_detection(path, allow_single_frame=True)
            self.assertEqual(3, len(trays))


if __name__ == "__main__":
    unittest.main()
