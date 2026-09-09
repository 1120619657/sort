import numpy as np

from sort import Sort


EMPTY = np.empty((0, 5))


def _det(x1, y1, x2, y2, score=0.9):
    return np.array([[x1, y1, x2, y2, score]], dtype=float)


def test_sort_single_target_continuous_motion_keeps_id():
    tracker = Sort(max_age=1, min_hits=1)
    ids = [int(tracker.update(_det(x, 0, x + 10, 10))[0, 4]) for x in (0, 1, 2)]
    assert ids == [1, 1, 1]


def test_sort_multiple_targets_are_returned():
    tracker = Sort(max_age=1, min_hits=1)
    detections = np.array([[0, 0, 10, 10, 0.9], [20, 20, 30, 30, 0.8]], dtype=float)
    result = tracker.update(detections)
    assert result.shape == (2, 5)
    assert set(result[:, 4].astype(int)) == {1, 2}


def test_sort_new_target_enters_with_new_id():
    tracker = Sort(max_age=1, min_hits=1)
    first = tracker.update(_det(0, 0, 10, 10))
    second = tracker.update(np.array([[1, 0, 11, 10, 0.9], [30, 30, 40, 40, 0.9]]))
    third = tracker.update(np.array([[2, 0, 12, 10, 0.9], [31, 30, 41, 40, 0.9]]))
    assert int(first[0, 4]) == 1
    assert set(second[:, 4].astype(int)) == {1}
    assert set(third[:, 4].astype(int)) == {1, 2}


def test_sort_target_leaves_current_output():
    tracker = Sort(max_age=2, min_hits=1)
    tracker.update(_det(0, 0, 10, 10))
    result = tracker.update(EMPTY)
    assert result.shape == (0, 5)
    assert len(tracker.trackers) == 1
