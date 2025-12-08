
import unittest

import numpy as np
import numpy.testing as npt

from pyrlutils.state.discretecoords import Discrete2DCartesianState


class Test2DDiscreteState(unittest.TestCase):
    def test_twobythree(self):
        state = Discrete2DCartesianState(0, 1, 0, 2, terminal_state_values=[(1, 2)])

        assert state.state_space_size == 6

        npt.assert_array_equal(state.state_value, np.array([0, 0], dtype=np.int64))

        state.set_state_value([1, 2])
        npt.assert_array_equal(state.state_value, np.array([1, 2], dtype=np.int64))

        state.set_state_value((0, 1))
        npt.assert_array_equal(state.state_value, np.array([0, 1], dtype=np.int64))

        assert state.get_whether_terminal_given_coordinates(np.array([1, 2], dtype=np.int64))


if __name__ == '__main__':
    unittest.main()
