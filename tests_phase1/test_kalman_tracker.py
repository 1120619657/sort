import numpy as np

from sort import KalmanBoxTracker


INITIAL_BOX = np.array([0.0, 0.0, 10.0, 20.0, 0.9])


def test_tracker_creation_initializes_state():
    tracker = KalmanBoxTracker(INITIAL_BOX)
    assert tracker.id == 0
    assert tracker.age == 0
    assert tracker.hits == 0
    assert tracker.hit_streak == 0
    assert tracker.time_since_update == 0
    np.testing.assert_allclose(tracker.get_state()[0], INITIAL_BOX[:4])


def test_tracker_predict_advances_age_and_time_since_update():
    tracker = KalmanBoxTracker(INITIAL_BOX)
    prediction = tracker.predict()
    assert prediction.shape == (1, 4)
    assert tracker.age == 1
    assert tracker.time_since_update == 1
    assert len(tracker.history) == 1


def test_tracker_update_resets_time_and_increments_hits():
    tracker = KalmanBoxTracker(INITIAL_BOX)
    tracker.predict()
    tracker.update(np.array([1.0, 0.0, 11.0, 20.0, 0.8]))
    assert tracker.time_since_update == 0
    assert tracker.hits == 1
    assert tracker.hit_streak == 1
    assert tracker.history == []


def test_tracker_hit_streak_resets_after_consecutive_miss():
    tracker = KalmanBoxTracker(INITIAL_BOX)
    tracker.update(INITIAL_BOX)
    tracker.predict()
    assert tracker.hit_streak == 1
    tracker.predict()
    assert tracker.hit_streak == 0


def test_tracker_get_state_returns_bbox_shape():
    tracker = KalmanBoxTracker(INITIAL_BOX)
    state = tracker.get_state()
    assert state.shape == (1, 4)
    assert np.all(np.isfinite(state))
