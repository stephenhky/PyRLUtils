
import unittest

from pyrlutils.state import DiscreteState


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


if __name__ == '__main__':
    unittest.main()
