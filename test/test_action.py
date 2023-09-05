
import unittest

from pyrlutils.state import DiscreteState
from pyrlutils.action import Action


class TestAction(unittest.TestCase):
    def test_deterministic_actions(self):
        state = DiscreteState([0, 1, 2])
        actions = {
            'left': Action(lambda state: state.set_state_value((state.state_value-1) % 3)),
            'right': Action(lambda state: state.set_state_value((state.state_value + 1) % 3)),
        }

        assert state.state_value == 0
        actions['left'](state)
        assert state.state_value == 2
        actions['left'](state)
        assert state.state_value == 1
        actions['left'](state)
        assert state.state_value == 0
        actions['right'](state)
        assert state.state_value == 1
        actions['right'](state)
        assert state.state_value == 2
        actions['right'](state)
        assert state.state_value == 0


if __name__ == '__main__':
    unittest.main()
