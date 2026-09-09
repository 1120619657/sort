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


def test_association_one_detection_matches_one_tracker():
    matches, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
        np.array([_box(0, 0, 10, 10)]), np.array([_box(0, 0, 10, 10)])
    )
    np.testing.assert_array_equal(matches, [[0, 0]])
    assert unmatched_dets.size == 0
    assert unmatched_trks.size == 0


def test_association_one_detection_does_not_match_distant_tracker():
    matches, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
        np.array([_box(0, 0, 10, 10)]), np.array([_box(20, 20, 30, 30)])
    )
    assert matches.shape == (0, 2)
    np.testing.assert_array_equal(unmatched_dets, [0])
    np.testing.assert_array_equal(unmatched_trks, [0])


def test_association_multiple_detections_and_trackers():
    detections = np.array([_box(0, 0, 10, 10), _box(20, 20, 30, 30)])
    trackers = np.array([_box(20, 20, 30, 30), _box(0, 0, 10, 10)])
    matches, unmatched_dets, unmatched_trks = associate_detections_to_trackers(detections, trackers)
    assert set(map(tuple, matches)) == {(0, 1), (1, 0)}
    assert unmatched_dets.size == 0
    assert unmatched_trks.size == 0


def test_association_more_detections_than_trackers():
    detections = np.array([_box(0, 0, 10, 10), _box(20, 20, 30, 30)])
    trackers = np.array([_box(0, 0, 10, 10)])
    matches, unmatched_dets, unmatched_trks = associate_detections_to_trackers(detections, trackers)
    np.testing.assert_array_equal(matches, [[0, 0]])
    np.testing.assert_array_equal(unmatched_dets, [1])
    assert unmatched_trks.size == 0


def test_association_more_trackers_than_detections():
    detections = np.array([_box(0, 0, 10, 10)])
    trackers = np.array([_box(0, 0, 10, 10), _box(20, 20, 30, 30)])
    matches, unmatched_dets, unmatched_trks = associate_detections_to_trackers(detections, trackers)
    np.testing.assert_array_equal(matches, [[0, 0]])
    assert unmatched_dets.size == 0
    np.testing.assert_array_equal(unmatched_trks, [1])


def test_association_iou_equal_to_threshold_is_accepted():
    detections = np.array([_box(0, 0, 2, 1)])
    trackers = np.array([_box(0, 0, 1, 1)])
    matches, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
        detections, trackers, iou_threshold=0.5
    )
    np.testing.assert_array_equal(matches, [[0, 0]])
    assert unmatched_dets.size == 0
    assert unmatched_trks.size == 0
