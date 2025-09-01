
import unittest

import numpy as np
import numpy.testing as npt

from pyrlutils.state import DiscreteState, ContinuousState, Discrete2DCartesianState
from pyrlutils.helpers.exceptions import InvalidRangeError


class TestState(unittest.TestCase):
    def test_discrete_state(self):
        all_values = ['a', 'b', 'c', 'd']
        state = DiscreteState(all_values.copy())

        assert state.state_value == all_values[0]
        assert state.get_state_value() == state.state_value
        assert state.get_all_possible_state_values()[2] == all_values[2]

        state.set_state_value(all_values[3])
        assert state.state_value == all_values[3]
        assert state.get_state_value() == state.state_value

        self.assertRaises(ValueError, state.set_state_value, 'e')

    def test_1d_continuous_state(self):
        state = ContinuousState(1, np.array([0.0, 1.0]))
        assert state.get_state_value() == state.state_value
        assert state.ranges[0, 0] == 0.0
        assert state.ranges[0, 1] == 1.0

        self.assertRaises(InvalidRangeError, ContinuousState, 1, np.array([0.0, 1.0]), 1.2)

    def test_2d_continuous_state(self):
        state = ContinuousState(2, np.array([[0.0, 1.0], [-1.0, 1.0]]))
        npt.assert_almost_equal(state.get_state_value(), state.state_value)
        assert state.ranges[0, 0] == 0.0
        assert state.ranges[0, 1] == 1.0
        assert state.ranges[1, 0] == -1.0
        assert state.ranges[1, 1] == 1.0

        self.assertRaises(InvalidRangeError, ContinuousState, 2, np.array([[0.0, 1.0], [-1.0, 1.0]]),
                          np.array([0.5, -1.3]))
        self.assertRaises(InvalidRangeError, ContinuousState, 2, np.array([[0.0, 1.0], [-1.0, 1.0]]),
                          np.array([1.5, -0.9]))
        self.assertRaises(InvalidRangeError, ContinuousState, 2, np.array([[0.0, 1.0], [-1.0, 1.0]]),
                          np.array([2.5, -1.4]))

    def test_2d_discrete_state(self):
        state = Discrete2DCartesianState(-5, 5, -3, 3, [0, 0])
        assert state.encode_coordinates([-5, -3]) == 0
        assert state.decode_coordinates(0) == [-5, -3]
        assert state.encode_coordinates([-4, -3]) == 1
        assert state.encode_coordinates([-5, -2]) == 11
        assert state.encode_coordinates([-5, -1]) == 22
        assert state.encode_coordinates([-5, 0]) == 33
        assert state.encode_coordinates([-5, 1]) == 44
        assert state.encode_coordinates([0, 1]) == 49
        assert state.decode_coordinates(38) == [0, 0]



if __name__ == '__main__':
    unittest.main()
