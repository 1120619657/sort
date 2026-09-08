import numpy as np

from sort import linear_assignment


def test_linear_assignment_single_pair():
    result = linear_assignment(np.array([[2.0]]))
    np.testing.assert_array_equal(result, np.array([[0, 0]]))


def test_linear_assignment_square_matrix_uses_minimum_total_cost():
    costs = np.array([[4.0, 1.0, 3.0], [2.0, 0.0, 5.0], [3.0, 2.0, 2.0]])
    result = linear_assignment(costs)
    assert set(map(tuple, result)) == {(0, 1), (1, 0), (2, 2)}


def test_linear_assignment_rectangular_matrix():
    costs = np.array([[1.0, 10.0, 10.0], [10.0, 1.0, 10.0]])
    result = linear_assignment(costs)
    assert set(map(tuple, result)) == {(0, 0), (1, 1)}


def test_linear_assignment_empty_matrix_has_pair_shape():
    result = linear_assignment(np.empty((0, 0)))
    assert result.shape == (0, 2)
