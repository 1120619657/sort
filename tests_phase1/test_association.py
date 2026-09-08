import numpy as np

from sort import associate_detections_to_trackers


def _box(x1, y1, x2, y2, score=1.0):
    return [x1, y1, x2, y2, score]


def test_association_no_detections_and_no_trackers():
    matches, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
        np.empty((0, 5)), np.empty((0, 5))
    )
    assert matches.shape == (0, 2)
    assert unmatched_dets.shape == (0,)
    assert unmatched_trks.shape == (0,)


def test_association_one_detection_and_no_trackers():
    matches, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
        np.array([_box(0, 0, 10, 10)]), np.empty((0, 5))
    )
    assert matches.shape == (0, 2)
    np.testing.assert_array_equal(unmatched_dets, [0])
    assert unmatched_trks.size == 0
