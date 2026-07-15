import importlib.util
from pathlib import Path
from types import SimpleNamespace


MODULE_PATH = Path(__file__).resolve().parents[1] / "scene3_locked_target_vision.py"
SPEC = importlib.util.spec_from_file_location("scene3_locked_target_vision", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def candidate(xyz, confidence=0.5, area=(10, 10)):
    return {
        "world_point": SimpleNamespace(
            point=SimpleNamespace(x=xyz[0], y=xyz[1], z=xyz[2]),
        ),
        "confidence": confidence,
        "bbox": (0, 0, area[0], area[1]),
    }


def test_nearest_identity_wins_over_confidence():
    wanted = candidate((1.50, -0.44, 1.16), confidence=0.40)
    wrong = candidate((1.51, -0.06, 1.17), confidence=0.99)
    selected, distance, reason = MODULE.select_locked_candidate(
        [wrong, wanted],
        (1.52, -0.442, 1.164),
        0.12,
    )
    assert selected is wanted
    assert distance < 0.03
    assert reason == "ok"


def test_every_candidate_can_be_selected_by_its_dynamic_lock():
    trays = [
        candidate((1.50, -0.44, 1.16)),
        candidate((1.50, -0.06, 1.16)),
        candidate((1.50, 0.32, 1.16)),
        candidate((1.50, -0.25, 0.72)),
        candidate((1.50, 0.13, 0.72)),
    ]
    for expected in trays:
        lock = MODULE.candidate_world_xyz(expected)
        selected, distance, reason = MODULE.select_locked_candidate(
            trays,
            lock,
            0.12,
        )
        assert selected is expected
        assert distance == 0.0
        assert reason == "ok"


def test_rejects_adjacent_tray_outside_gate():
    wrong = candidate((1.516, -0.059, 1.170), confidence=0.99)
    selected, distance, reason = MODULE.select_locked_candidate(
        [wrong],
        (1.520, -0.442, 1.164),
        0.12,
    )
    assert selected is None
    assert distance > 0.38
    assert reason == "nearest candidate is outside identity gate"


def test_rejects_missing_or_invalid_lock():
    tray = candidate((1.0, 2.0, 3.0))
    for lock in (None, [], [1.0, 2.0], [1.0, 2.0, float("nan")]):
        selected, distance, reason = MODULE.select_locked_candidate(
            [tray],
            lock,
            0.12,
        )
        assert selected is None
        assert distance is None
        assert reason == "invalid target lock"


def test_rejects_candidates_without_world_coordinates():
    selected, distance, reason = MODULE.select_locked_candidate(
        [{"confidence": 1.0, "bbox": (0, 0, 100, 100)}],
        (1.0, 2.0, 3.0),
        0.12,
    )
    assert selected is None
    assert distance is None
    assert reason == "no candidate has a world coordinate"


def test_confidence_breaks_exact_distance_tie():
    low = candidate((1.0, 2.0, 3.0), confidence=0.4)
    high = candidate((1.0, 2.0, 3.0), confidence=0.8)
    selected, distance, reason = MODULE.select_locked_candidate(
        [low, high],
        (1.0, 2.0, 3.0),
        0.12,
    )
    assert selected is high
    assert distance == 0.0
    assert reason == "ok"
