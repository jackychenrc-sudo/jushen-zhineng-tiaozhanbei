#!/usr/bin/env python3

import os
import sys
import unittest


HERE = os.path.dirname(os.path.abspath(__file__))
SERVER = os.path.dirname(HERE)
if SERVER not in sys.path:
    sys.path.insert(0, SERVER)

from scene3_safe_turn_position import build_parser, run_ros  # noqa: E402


class RetiredSafeTurnPositionTest(unittest.TestCase):

    def test_old_entry_point_is_fail_closed_even_with_execute_flag(self):
        args = build_parser().parse_args([
            "--execute", "--confirmation", "SCENE3_SAFE_TURN_POSITION"
        ])
        self.assertEqual(run_ros(args), 2)


if __name__ == "__main__":
    unittest.main()
