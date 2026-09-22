"""Phase 2 / Checkpoint 4: AI-generated complex and long-sequence tests."""

import numpy as np
import pytest

from sort import Sort


pytestmark = [pytest.mark.ai_generated, pytest.mark.ai_scenario]
EMPTY = np.empty((0, 5))


def _det(x, y=0.0, w=20.0, h=20.0, score=0.9):
    return np.array([[x, y, x + w, y + h, score]], dtype=float)


def test_ai_500_frame_smooth_motion_keeps_single_id():
    tracker = Sort(max_age=2, min_hits=1, iou_threshold=0.3)
    seen_ids = []
    for frame in range(500):
        result = tracker.update(_det(frame * 0.25))
        assert result.shape == (1, 5)
        seen_ids.append(int(result[0, 4]))
        assert len(tracker.trackers) == 1
    assert len(set(seen_ids)) == 1


def test_ai_500_frame_periodic_single_miss_preserves_track():
    tracker = Sort(max_age=2, min_hits=1, iou_threshold=0.3)
    first_id = None
    observed_ids = []
    for frame in range(500):
        if frame > 0 and frame % 25 == 0:
            assert tracker.update(EMPTY).shape == (0, 5)
            continue
        result = tracker.update(_det(frame * 0.2))
        assert result.shape == (1, 5)
        current_id = int(result[0, 4])
        first_id = current_id if first_id is None else first_id
        observed_ids.append(current_id)
        assert len(tracker.trackers) == 1
    assert set(observed_ids) == {first_id}


def test_ai_two_targets_cross_without_duplicate_output_ids_or_tracker_explosion():
    tracker = Sort(max_age=2, min_hits=1, iou_threshold=0.2)
    for frame in range(80):
        left_to_right = frame * 0.5
        right_to_left = 40.0 - frame * 0.5
        detections = np.array(
            [
                [left_to_right, 0.0, left_to_right + 12.0, 12.0, 0.9],
                [right_to_left, 0.0, right_to_left + 12.0, 12.0, 0.9],
            ],
            dtype=float,
        )
        result = tracker.update(detections)
        ids = result[:, 4].astype(int).tolist()
        assert len(ids) == len(set(ids))
        assert len(tracker.trackers) <= 4


def test_ai_sudden_large_displacement_creates_new_track_instead_of_false_match():
    tracker = Sort(max_age=2, min_hits=1, iou_threshold=0.3)
    first = tracker.update(_det(0.0))
    first_id = int(first[0, 4])

    # No geometric overlap with the old track.
    second = tracker.update(_det(100.0))
    assert second.shape == (0, 5)
    assert len(tracker.trackers) == 2

    third = tracker.update(_det(100.5))
    assert third.shape == (1, 5)
    assert int(third[0, 4]) != first_id


def test_ai_repeated_entry_and_expiry_does_not_leak_trackers():
    tracker = Sort(max_age=1, min_hits=1, iou_threshold=0.3)
    for cycle in range(100):
        tracker.update(_det(float(cycle * 100)))
        tracker.update(EMPTY)
        tracker.update(EMPTY)
        assert len(tracker.trackers) == 0
