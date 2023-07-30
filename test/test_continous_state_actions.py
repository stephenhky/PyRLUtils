
import unittest

import numpy as np

from pyrlutils.state import ContinuousState


class TestContinuousSystem(unittest.TestCase):
    def test_1d(self):
        state = ContinuousState(1, np.array([-1.5, 1.5]), init_value=0.0)
        actions = {
            'increase': lambda s, v: s.set_state_value(s.state_value+v if s.state_value < 1.5-v else 1.5),
            'decrease': lambda s, v: s.set_state_value(s.state_value-v if s.state_value > -1.5+v else -1.5)
        }

        assert round(state.state_value, 1) == 0.0
        actions['increase'](state, 0.2)
        assert round(state.state_value, 1) == 0.2
        actions['increase'](state, 1.0)
        assert round(state.state_value, 1) == 1.2
        actions['increase'](state, 1.0)
        assert round(state.state_value, 1) == 1.5
        actions['decrease'](state, 0.4)
        assert round(state.state_value, 1) == 1.1
        actions['decrease'](state, 2.0)
        print(state.state_value)
        assert round(state.state_value, 1) == -0.9
        actions['decrease'](state, 1.0)
        assert round(state.state_value, 1) == -1.5
        actions['increase'](state, 0.5)
        assert round(state.state_value, 1) == -1.0
        state.state_value = 0.0
        assert round(state.state_value, 1) == 0.0


if __name__ == '__main__':
    unittest.main()
