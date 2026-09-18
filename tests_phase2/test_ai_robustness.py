"""Phase 2 / Checkpoint 1: AI-generated abnormal-input and numerical tests.

The oracle used for abnormal numerical input is intentionally conservative:
a function may either reject the input explicitly with ValueError, or return a
finite numeric result. Silent NaN/Inf propagation is treated as a robustness
failure and must be reviewed manually before it is classified as a defect.
"""

import numpy as np
import pytest

from sort import convert_bbox_to_z, convert_x_to_bbox, iou_batch


pytestmark = [pytest.mark.ai_generated, pytest.mark.ai_robustness]


def _finite_or_value_error(callable_):
    """Accept explicit validation or a finite numeric result, but not NaN/Inf."""
    try:
        result = callable_()
    except ValueError:
        return
    assert np.all(np.isfinite(np.asarray(result, dtype=float)))


def test_ai_zero_width_self_iou_is_not_silent_nan():
    box = np.array([[5.0, 0.0, 5.0, 10.0]])
    _finite_or_value_error(lambda: iou_batch(box, box))


def test_ai_zero_height_bbox_conversion_is_not_silent_inf():
    bbox = np.array([0.0, 4.0, 10.0, 4.0])
    _finite_or_value_error(lambda: convert_bbox_to_z(bbox))


def test_ai_nan_coordinate_is_rejected_or_kept_finite():
    box = np.array([[0.0, 0.0, np.nan, 10.0]])
    reference = np.array([[0.0, 0.0, 10.0, 10.0]])
    _finite_or_value_error(lambda: iou_batch(box, reference))


def test_ai_infinite_coordinate_is_rejected_or_kept_finite():
    box = np.array([[0.0, 0.0, np.inf, 10.0]])
    reference = np.array([[0.0, 0.0, 10.0, 10.0]])
    _finite_or_value_error(lambda: iou_batch(box, reference))


def test_ai_very_large_finite_bbox_round_trip_stays_finite():
    bbox = np.array([1.0e12, -1.0e12, 1.0e12 + 64.0, -1.0e12 + 32.0])
    restored = convert_x_to_bbox(convert_bbox_to_z(bbox))[0]
    assert np.all(np.isfinite(restored))
    np.testing.assert_allclose(restored, bbox, rtol=0.0, atol=1e-6)


def test_ai_negative_coordinates_keep_iou_in_valid_range():
    a = np.array([[-10.0, -10.0, 0.0, 0.0]])
    b = np.array([[-5.0, -5.0, 5.0, 5.0]])
    value = float(iou_batch(a, b)[0, 0])
    assert np.isfinite(value)
    assert 0.0 <= value <= 1.0
