"""Phase 2 / Checkpoint 2: AI-generated parameter-combination tests."""

import numpy as np
import pytest

from sort import Sort


pytestmark = [pytest.mark.ai_generated, pytest.mark.ai_parameters]
EMPTY = np.empty((0, 5))


def _det(x1, y1, x2, y2, score=0.9):
    return np.array([[x1, y1, x2, y2, score]], dtype=float)


def test_ai_max_age_two_survives_exactly_two_misses_then_expires():
    tracker = Sort(max_age=2, min_hits=1)
    tracker.update(_det(0, 0, 10, 10))
    tracker.update(EMPTY)
    assert len(tracker.trackers) == 1
    tracker.update(EMPTY)
    assert len(tracker.trackers) == 1
    tracker.update(EMPTY)
    assert len(tracker.trackers) == 0


def test_ai_reappearance_on_last_allowed_miss_keeps_id():
    tracker = Sort(max_age=2, min_hits=1)
    first_id = int(tracker.update(_det(0, 0, 10, 10))[0, 4])
    tracker.update(EMPTY)
    tracker.update(EMPTY)
    result = tracker.update(_det(1, 0, 11, 10))
    assert result.shape == (1, 5)
    assert int(result[0, 4]) == first_id


def test_ai_reappearance_after_expiry_gets_new_id():
    tracker = Sort(max_age=2, min_hits=1)
    first_id = int(tracker.update(_det(0, 0, 10, 10))[0, 4])
    tracker.update(EMPTY)
    tracker.update(EMPTY)
    tracker.update(EMPTY)
    assert len(tracker.trackers) == 0

    # After the startup window, a newly created track is not returned until it
    # receives a subsequent associated detection. Inspect the internal ID first,
    # then verify that the next matching frame returns that new ID.
    assert tracker.update(_det(1, 0, 11, 10)).shape == (0, 5)
    assert len(tracker.trackers) == 1
    new_id = tracker.trackers[0].id + 1
    assert new_id != first_id

    result = tracker.update(_det(2, 0, 12, 10))
    assert result.shape == (1, 5)
    assert int(result[0, 4]) == new_id


def test_ai_iou_equal_threshold_keeps_track_at_system_level():
    tracker = Sort(max_age=1, min_hits=1, iou_threshold=0.5)
    first_id = int(tracker.update(_det(0, 0, 1, 1))[0, 4])
    # Intersection=1, union=2 -> IoU=0.5 exactly.
    result = tracker.update(_det(0, 0, 2, 1))
    assert result.shape == (1, 5)
    assert int(result[0, 4]) == first_id
    assert len(tracker.trackers) == 1


def test_ai_iou_just_below_required_threshold_creates_second_track():
    tracker = Sort(max_age=2, min_hits=1, iou_threshold=0.500001)
    tracker.update(_det(0, 0, 1, 1))
    result = tracker.update(_det(0, 0, 2, 1))
    # IoU is exactly 0.5, so it must not be associated at a slightly higher threshold.
    assert result.shape == (0, 5)
    assert len(tracker.trackers) == 2


def test_ai_min_hits_rebuilds_streak_after_a_miss():
    tracker = Sort(max_age=2, min_hits=2)
    # Move beyond the startup shortcut (frame_count <= min_hits).
    tracker.update(EMPTY)
    tracker.update(EMPTY)

    assert tracker.update(_det(0, 0, 10, 10)).shape == (0, 5)
    assert tracker.update(_det(1, 0, 11, 10)).shape == (0, 5)
    ready = tracker.update(_det(2, 0, 12, 10))
    assert ready.shape == (1, 5)
    original_id = int(ready[0, 4])

    tracker.update(EMPTY)
    assert tracker.update(_det(3, 0, 13, 10)).shape == (0, 5)
    recovered = tracker.update(_det(4, 0, 14, 10))
    assert recovered.shape == (1, 5)
    assert int(recovered[0, 4]) == original_id
