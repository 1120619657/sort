"""Phase 2 / Checkpoint 3: AI-derived mathematical/property tests.

Hypothesis generates many valid bounding boxes for each collected pytest case.
The five tests below are counted as five Phase 2 cases even though each test
executes multiple generated examples internally.
"""

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from sort import (
    associate_detections_to_trackers,
    convert_bbox_to_z,
    convert_x_to_bbox,
    iou_batch,
)


pytestmark = [pytest.mark.ai_generated, pytest.mark.ai_property]


@st.composite
def valid_bbox(draw):
    x = draw(st.integers(min_value=-1000, max_value=1000))
    y = draw(st.integers(min_value=-1000, max_value=1000))
    w = draw(st.integers(min_value=1, max_value=200))
    h = draw(st.integers(min_value=1, max_value=200))
    return (float(x), float(y), float(x + w), float(y + h))


AI_SETTINGS = settings(max_examples=60, deadline=None)


@AI_SETTINGS
@given(a=valid_bbox(), b=valid_bbox())
def test_ai_property_iou_is_symmetric(a, b):
    a_arr = np.array([a], dtype=float)
    b_arr = np.array([b], dtype=float)
    ab = float(iou_batch(a_arr, b_arr)[0, 0])
    ba = float(iou_batch(b_arr, a_arr)[0, 0])
    assert ab == pytest.approx(ba, rel=1e-12, abs=1e-12)


@AI_SETTINGS
@given(box=valid_bbox())
def test_ai_property_self_iou_is_one_for_positive_area_box(box):
    arr = np.array([box], dtype=float)
    value = float(iou_batch(arr, arr)[0, 0])
    assert value == pytest.approx(1.0, rel=1e-12, abs=1e-12)


@AI_SETTINGS
@given(a=valid_bbox(), b=valid_bbox())
def test_ai_property_iou_stays_in_unit_interval(a, b):
    value = float(iou_batch(np.array([a]), np.array([b]))[0, 0])
    assert np.isfinite(value)
    assert 0.0 <= value <= 1.0


@AI_SETTINGS
@given(box=valid_bbox())
def test_ai_property_bbox_round_trip_preserves_valid_box(box):
    original = np.array(box, dtype=float)
    restored = convert_x_to_bbox(convert_bbox_to_z(original))[0]
    np.testing.assert_allclose(restored, original, rtol=1e-10, atol=1e-10)


@settings(max_examples=40, deadline=None)
@given(
    detections=st.lists(valid_bbox(), min_size=1, max_size=5),
    trackers=st.lists(valid_bbox(), min_size=1, max_size=5),
)
def test_ai_property_association_is_one_to_one(detections, trackers):
    dets = np.array(detections, dtype=float)
    trks = np.array(trackers, dtype=float)
    matches, _, _ = associate_detections_to_trackers(dets, trks, iou_threshold=0.3)

    det_indices = matches[:, 0].tolist() if len(matches) else []
    trk_indices = matches[:, 1].tolist() if len(matches) else []
    assert len(det_indices) == len(set(det_indices))
    assert len(trk_indices) == len(set(trk_indices))
