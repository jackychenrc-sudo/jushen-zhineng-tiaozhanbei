import sys
import unittest
from argparse import Namespace
from pathlib import Path


SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

import scene3_multiframe_stability as stability


def settings(**overrides):
    values = {
        "expected_count": 3,
        "minimum_frames": 3,
        "minimum_selection_score": 0.60,
        "minimum_depth_shape_score": 0.75,
        "maximum_pixel_spread": 12.0,
        "maximum_position_spread": 0.02,
    }
    values.update(overrides)
    return Namespace(**values)


def tray(name, u, xyz, v=245, geometry="rgbd_vertical", accepted=True):
    object_bbox = [u - 8, v - 25, u + 8, v + 25]
    return {
        "id": name,
        "accepted": accepted,
        "geometry_source": geometry,
        "proposal_sources": ["multiscale_template", "rgbd_vertical"],
        "object_bbox": object_bbox,
        "bbox": object_bbox,
        "detection_bbox": [u - 7, v - 24, u + 7, v + 24],
        "object_center_pixel": [u, v],
        "center_pixel": [u, v],
        "depth_m": xyz[0],
        "camera_xyz_m": [xyz[1], -0.4, xyz[0]],
        "base_link_xyz_m": list(xyz),
        "base_link_xyz_raw_m": list(xyz),
        "base_link_xyz_corrected_m": [xyz[0] + 0.001, xyz[1], xyz[2]],
        "selection_score": 0.82,
        "depth_shape_score": 0.90,
        "template_score": 0.55,
    }


def frame(items, source_frame="Head Camera View", target_frame="base_link"):
    return {
        "status": "ok",
        "algorithm": "multiscale_rgbd_object_geometry_v5",
        "source_frame": source_frame,
        "target_frame": target_frame,
        "upper_trays": items,
    }


class MultiframeStabilityTests(unittest.TestCase):
    def stable_frames(self):
        first = [
            tray("a", 500, [1.300, 0.42, 0.270]),
            tray("b", 650, [1.302, -0.02, 0.272]),
            tray("c", 760, [1.299, -0.36, 0.271]),
        ]
        second = [
            tray("c", 762, [1.303, -0.359, 0.273], v=246),
            tray("a", 498, [1.296, 0.421, 0.268], v=244),
            tray("b", 651, [1.304, -0.021, 0.271], v=246),
        ]
        third = [
            tray("b", 649, [1.301, -0.019, 0.274], v=244),
            tray("c", 759, [1.298, -0.361, 0.270], v=245),
            tray("a", 501, [1.302, 0.419, 0.271], v=246),
        ]
        return [frame(first), frame(second), frame(third)]

    def test_stable_jitter_and_order_changes_pass(self):
        payload = stability.build_consensus(
            self.stable_frames(), [Path("a"), Path("b"), Path("c")], settings()
        )

        self.assertEqual("ok", payload["status"])
        self.assertTrue(payload["temporal_validation"]["passed"])
        self.assertEqual([500, 650, 760], [t["center_pixel"][0] for t in payload["upper_trays"]])
        self.assertEqual(
            ["upper_x0", "upper_x1", "upper_x2"],
            [t["id"] for t in payload["upper_trays"]],
        )
        self.assertTrue(all(t["temporal_metrics"]["stable"] for t in payload["upper_trays"]))
        self.assertAlmostEqual(1.300, payload["upper_trays"][0]["base_link_xyz_raw_m"][0])

    def test_large_position_jump_is_unstable(self):
        frames = self.stable_frames()
        frames[2]["upper_trays"][2]["base_link_xyz_raw_m"][0] += 0.04
        frames[2]["upper_trays"][2]["base_link_xyz_m"][0] += 0.04

        payload = stability.build_consensus(frames, [], settings())

        self.assertEqual("unstable", payload["status"])
        self.assertFalse(payload["temporal_validation"]["passed"])
        self.assertEqual([], payload["upper_trays"])
        self.assertTrue(any(not track["temporal_metrics"]["stable"] for track in payload["tracks"]))

    def test_count_mismatch_is_rejected(self):
        frames = self.stable_frames()
        frames[1]["upper_trays"].pop()

        payload = stability.build_consensus(frames, [], settings())

        self.assertEqual("rejected_input", payload["status"])
        self.assertTrue(any("count is 2" in error for error in payload["temporal_validation"]["errors"]))

    def test_template_only_geometry_is_rejected(self):
        frames = self.stable_frames()
        frames[0]["upper_trays"][0]["geometry_source"] = "template"

        payload = stability.build_consensus(frames, [], settings())

        self.assertEqual("rejected_input", payload["status"])
        self.assertTrue(any("lacks RGB-D geometry" in error for error in payload["temporal_validation"]["errors"]))

    def test_coordinate_frame_change_is_rejected(self):
        frames = self.stable_frames()
        frames[2]["target_frame"] = "odom"

        payload = stability.build_consensus(frames, [], settings())

        self.assertEqual("rejected_input", payload["status"])
        self.assertTrue(any("target coordinate frame changed" in error for error in payload["temporal_validation"]["errors"]))


if __name__ == "__main__":
    unittest.main()
