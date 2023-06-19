
import unittest

import numpy as np

from pyrlutils.state import DiscreteState, ContinuousState, InvalidRangeError


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



if __name__ == '__main__':
    unittest.main()
