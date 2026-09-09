import numpy as np
import pytest

from sort import convert_bbox_to_z, convert_x_to_bbox


@pytest.mark.parametrize(
    ("bbox", "expected"),
    [
        ([0.0, 0.0, 10.0, 20.0], [5.0, 10.0, 200.0, 0.5]),
        ([2.0, 3.0, 12.0, 13.0], [7.0, 8.0, 100.0, 1.0]),
        ([0.0, 0.0, 20.0, 5.0], [10.0, 2.5, 100.0, 4.0]),
        ([0.0, 0.0, 5.0, 20.0], [2.5, 10.0, 100.0, 0.25]),
        ([0.25, 1.5, 3.75, 6.5], [2.0, 4.0, 17.5, 0.7]),
    ],
)
def test_convert_bbox_to_z_known_rectangles(bbox, expected):
    result = convert_bbox_to_z(np.array(bbox))
    assert result.shape == (4, 1)
    np.testing.assert_allclose(result.ravel(), expected)


def test_convert_x_to_bbox_without_score():
    state = np.array([5.0, 10.0, 200.0, 0.5])
    result = convert_x_to_bbox(state)
    assert result.shape == (1, 4)
    np.testing.assert_allclose(result[0], [0.0, 0.0, 10.0, 20.0])


def test_convert_x_to_bbox_with_score():
    state = np.array([5.0, 10.0, 200.0, 0.5])
    result = convert_x_to_bbox(state, score=0.85)
    assert result.shape == (1, 5)
    np.testing.assert_allclose(result[0], [0.0, 0.0, 10.0, 20.0, 0.85])


def test_bbox_conversion_round_trip():
    bbox = np.array([-3.5, 2.25, 8.75, 19.5])
    result = convert_x_to_bbox(convert_bbox_to_z(bbox))[0]
    np.testing.assert_allclose(result, bbox, rtol=1e-12, atol=1e-12)
