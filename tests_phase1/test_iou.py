import numpy as np
import pytest

from sort import iou_batch


def test_iou_identical_boxes_is_one():
    result = iou_batch(np.array([[0.0, 0.0, 2.0, 2.0]]), np.array([[0.0, 0.0, 2.0, 2.0]]))
    assert result[0, 0] == pytest.approx(1.0)


def test_iou_disjoint_boxes_is_zero():
    result = iou_batch(np.array([[0.0, 0.0, 1.0, 1.0]]), np.array([[2.0, 2.0, 3.0, 3.0]]))
    assert result[0, 0] == pytest.approx(0.0)


def test_iou_partial_overlap():
    result = iou_batch(np.array([[0.0, 0.0, 2.0, 2.0]]), np.array([[1.0, 1.0, 3.0, 3.0]]))
    assert result[0, 0] == pytest.approx(1.0 / 7.0)


def test_iou_contained_box():
    result = iou_batch(np.array([[1.0, 1.0, 3.0, 3.0]]), np.array([[0.0, 0.0, 4.0, 4.0]]))
    assert result[0, 0] == pytest.approx(0.25)


def test_iou_touching_edges_is_zero():
    result = iou_batch(np.array([[0.0, 0.0, 1.0, 1.0]]), np.array([[1.0, 0.0, 2.0, 1.0]]))
    assert result[0, 0] == pytest.approx(0.0)
