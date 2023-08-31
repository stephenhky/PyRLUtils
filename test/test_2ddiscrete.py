
import unittest

from pyrlutils.state import Discrete2DCartesianState


class Test2DDiscreteState(unittest.TestCase):
    def test_twobythree(self):
        state = Discrete2DCartesianState(0, 1, 0, 2)

        assert state.state_space_size == 6
        assert state.state_value == 0

        state.set_state_value(5)
        assert state.decode_coordinates(state.state_value) == [1, 2]



if __name__ == '__main__':
    unittest.main()
