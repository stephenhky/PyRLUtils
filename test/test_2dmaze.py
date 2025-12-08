
from itertools import product
import unittest

import numpy as np

from pyrlutils.transition import TransitionProbabilityFactory, NextStateTuple
from pyrlutils.dp.valuefcns import OptimalPolicyOnValueFunctions
from pyrlutils.state.discretecoords import Discrete2DCartesianState
from pyrlutils.state.discrete import DiscreteCategoricalState


class Test2DMaze(unittest.TestCase):
    def setUp(self):
        maze_state = Discrete2DCartesianState(0, 5, 0, 4,
                                              initial_coordinate=np.array([0, 0], dtype=np.int64))

        transprobfactory = TransitionProbabilityFactory()
        transprobfactory.add_state_transitions(
            np.array([0, 0], dtype=np.int64),
            {
                'up': [NextStateTuple(np.array([0, 0], dtype=np.int64), 1., 0., False)],
                'down': [NextStateTuple(np.array([0, 0], dtype=np.int64), 1., 0., False)],
                'left': [NextStateTuple(np.array([0, 0], dtype=np.int64), 1., 0., False)],
                'right': [NextStateTuple(np.array([0, 1], dtype=np.int64), 1., 0., False)]
            }
        )
        transprobfactory.add_state_transitions(
            np.array([0, 1], dtype=np.int64),
            {
                'up': [NextStateTuple(np.array([0, 1], dtype=np.int64), 1., 0., False)],
                'down': [NextStateTuple(np.array([1, 1], dtype=np.int64), 1., 0., False)],
                'left': [NextStateTuple(np.array([0, 0], dtype=np.int64), 1., 0., False)],
                'right': [NextStateTuple(np.array([0, 2], dtype=np.int64), 1., 0., False)]
            }
        )
        transprobfactory.add_state_transitions(
            np.array([0, 2], dtype=np.int64),
            {
                'up': [NextStateTuple(np.array([0, 2], dtype=np.int64), 1., 0., False)],
                'down': [NextStateTuple(np.array([0, 2], dtype=np.int64), 1., 0., False)],
                'left': [NextStateTuple(np.array([0, 1], dtype=np.int64), 1., 0., False)],
                'right': [NextStateTuple(np.array([0, 3], dtype=np.int64), 1., 0., False)]
            }
        )
        transprobfactory.add_state_transitions(
            np.array([0, 3], dtype=np.int64),
            {
                'up': [NextStateTuple(np.array([0, 3], dtype=np.int64), 1., 0., False)],
                'down': [NextStateTuple(np.array([1, 3], dtype=np.int64), 1., 0., False)],
                'left': [NextStateTuple(np.array([0, 2], dtype=np.int64), 1., 0., False)],
                'right': [NextStateTuple(np.array([0, 4], dtype=np.int64), 1., 0., False)]
            }
        )
        transprobfactory.add_state_transitions(
            np.array([0, 4], dtype=np.int64),
            {
                'up': [NextStateTuple(np.array([0, 4], dtype=np.int64), 1., 0., False)],
                'down': [NextStateTuple(np.array([1, 4], dtype=np.int64), 1., 0., False)],
                'left': [NextStateTuple(np.array([0, 3], dtype=np.int64), 1., 0., False)],
                'right': [NextStateTuple(np.array([0, 4], dtype=np.int64), 1., 0., False)]
            }
        )
        transprobfactory.add_state_transitions(
            np.array([1, 0], dtype=np.int64),
            {
                'up': [NextStateTuple(np.array([1, 0], dtype=np.int64), 1., 0., False)],
                'down': [NextStateTuple(np.array([2, 0], dtype=np.int64), 1., 0., False)],
                'left': [NextStateTuple(np.array([1, 0], dtype=np.int64), 1., 0., False)],
                'right': [NextStateTuple(np.array([1, 1], dtype=np.int64), 1., 0., False)]
            }
        )
        transprobfactory.add_state_transitions(
            np.array([1, 1], dtype=np.int64),
            {
                'up': [NextStateTuple(np.array([0, 1], dtype=np.int64), 1., 0., False)],
                'down': [NextStateTuple(np.array([1, 1], dtype=np.int64), 1., 0., False)],
                'left': [NextStateTuple(np.array([1, 0], dtype=np.int64), 1., 0., False)],
                'right': [NextStateTuple(np.array([1, 1], dtype=np.int64), 1., 0., False)]
            }
        )
        transprobfactory.add_state_transitions(
            np.array([1, 2], dtype=np.int64),
            {
                'up': [NextStateTuple(np.array([1, 2], dtype=np.int64), 1., 0., False)],
                'down': [NextStateTuple(np.array([2, 2], dtype=np.int64), 1., 0., False)],
                'left': [NextStateTuple(np.array([1, 2], dtype=np.int64), 1., 0., False)],
                'right': [NextStateTuple(np.array([1, 3], dtype=np.int64), 1., 0., False)]
            }
        )
        transprobfactory.add_state_transitions(
            np.array([1, 3], dtype=np.int64),
            {
                'up': [NextStateTuple(np.array([0, 3], dtype=np.int64), 1., 0., False)],
                'down': [NextStateTuple(np.array([1, 3], dtype=np.int64), 1., 0., False)],
                'left': [NextStateTuple(np.array([1, 2], dtype=np.int64), 1., 0., False)],
                'right': [NextStateTuple(np.array([1, 4], dtype=np.int64), 1., 0., False)]
            }
        )
        transprobfactory.add_state_transitions(
            np.array([1, 4], dtype=np.int64),
            {
                'up': [NextStateTuple(np.array([0, 4], dtype=np.int64), 1., 0., False)],
                'down': [NextStateTuple(np.array([1, 4], dtype=np.int64), 1., 0., False)],
                'left': [NextStateTuple(np.array([1, 4], dtype=np.int64), 1., 0., False)],
                'right': [NextStateTuple(np.array([1, 4], dtype=np.int64), 1., 0., False)]
            }
        )
        transprobfactory.add_state_transitions(
            np.array([2, 0], dtype=np.int64),
            {
                'up': [NextStateTuple(np.array([1, 0], dtype=np.int64), 1., 0., False)],
                'down': [NextStateTuple(np.array([3, 0], dtype=np.int64), 1., 0., False)],
                'left': [NextStateTuple(np.array([2, 0], dtype=np.int64), 1., 0., False)],
                'right': [NextStateTuple(np.array([2, 1], dtype=np.int64), 1., 0., False)]
            }
        )
        transprobfactory.add_state_transitions(
            np.array([2, 1], dtype=np.int64),
            {
                'up': [NextStateTuple(np.array([2, 1], dtype=np.int64), 1., 0., False)],
                'down': [NextStateTuple(np.array([2, 1], dtype=np.int64), 1., 0., False)],
                'left': [NextStateTuple(np.array([2, 0], dtype=np.int64), 1., 0., False)],
                'right': [NextStateTuple(np.array([2, 2], dtype=np.int64), 1., 0., False)]
            }
        )
        transprobfactory.add_state_transitions(
            np.array([2, 2], dtype=np.int64),
            {
                'up': [NextStateTuple(np.array([1, 2], dtype=np.int64), 1., 0., False)],
                'down': [NextStateTuple(np.array([2, 2], dtype=np.int64), 1., 0., False)],
                'left': [NextStateTuple(np.array([2, 1], dtype=np.int64), 1., 0., False)],
                'right': [NextStateTuple(np.array([2, 3], dtype=np.int64), 1., 0., False)]
            }
        )
        transprobfactory.add_state_transitions(
            np.array([2, 3], dtype=np.int64),
            {
                'up': [NextStateTuple(np.array([2, 3], dtype=np.int64), 1., 0., False)],
                'down': [NextStateTuple(np.array([3, 3], dtype=np.int64), 1., 0., False)],
                'left': [NextStateTuple(np.array([2, 2], dtype=np.int64), 1., 0., False)],
                'right': [NextStateTuple(np.array([2, 3], dtype=np.int64), 1., 0., False)]
            }
        )
        transprobfactory.add_state_transitions(
            np.array([2, 4], dtype=np.int64),
            {
                'up': [NextStateTuple(np.array([2, 4], dtype=np.int64), 1., 0., False)],
                'down': [NextStateTuple(np.array([3, 4], dtype=np.int64), 1., 0., False)],
                'left': [NextStateTuple(np.array([2, 4], dtype=np.int64), 1., 0., False)],
                'right': [NextStateTuple(np.array([2, 4], dtype=np.int64), 1., 0., False)]
            }
        )
        transprobfactory.add_state_transitions(
            np.array([3, 0]),
            {
                'up': [NextStateTuple(np.array([2, 0], dtype=np.int64), 1., 0., False)],
                'down': [NextStateTuple(np.array([4, 0], dtype=np.int64), 1., 0., False)],
                'left': [NextStateTuple(np.array([3, 0], dtype=np.int64), 1., 0., False)],
                'right': [NextStateTuple(np.array([3, 1], dtype=np.int64), 1., 0., False)]
            }
        )
        transprobfactory.add_state_transitions(
            np.array([3, 1], dtype=np.int64),
            {
                'up': [NextStateTuple(np.array([3, 1], dtype=np.int64), 1., 0., False)],
                'down': [NextStateTuple(np.array([3, 1], dtype=np.int64), 1., 0., False)],
                'left': [NextStateTuple(np.array([3, 0], dtype=np.int64), 1., 0., False)],
                'right': [NextStateTuple(np.array([3, 2], dtype=np.int64), 1., 0., False)]
            }
        )
        transprobfactory.add_state_transitions(
            np.array([3, 2], dtype=np.int64),
            {
                'up': [NextStateTuple(np.array([3, 2], dtype=np.int64), 1., 0., False)],
                'down': [NextStateTuple(np.array([4, 2], dtype=np.int64), 1., 0., False)],
                'left': [NextStateTuple(np.array([3, 1], dtype=np.int64), 1., 0., False)],
                'right': [NextStateTuple(np.array([3, 3], dtype=np.int64), 1., 0., False)]
            }
        )
        transprobfactory.add_state_transitions(
            np.array([3, 3], dtype=np.int64),
            {
                'up': [NextStateTuple(np.array([2, 3], dtype=np.int64), 1., 0., False)],
                'down': [NextStateTuple(np.array([4, 3], dtype=np.int64), 1., 0., False)],
                'left': [NextStateTuple(np.array([3, 3], dtype=np.int64), 1., 0., False)],
                'right': [NextStateTuple(np.array([3, 4], dtype=np.int64), 1., 0., False)]
            }
        )
        transprobfactory.add_state_transitions(
            np.array([3, 4], dtype=np.int64),
            {
                'up': [NextStateTuple(np.array([2, 4], dtype=np.int64), 1., 0., False)],
                'down': [NextStateTuple(np.array([3, 4], dtype=np.int64), 1., 0., False)],
                'left': [NextStateTuple(np.array([3, 3], dtype=np.int64), 1., 0., False)],
                'right': [NextStateTuple(np.array([3, 4], dtype=np.int64), 1., 0., False)]
            }
        )
        transprobfactory.add_state_transitions(
            np.array([4, 0], dtype=np.int64),
            {
                'up': [NextStateTuple(np.array([3, 0], dtype=np.int64), 1., 0., False)],
                'down': [NextStateTuple(np.array([5, 0], dtype=np.int64), 1., 0., False)],
                'left': [NextStateTuple(np.array([4, 0], dtype=np.int64), 1., 0., False)],
                'right': [NextStateTuple(np.array([4, 0], dtype=np.int64), 1., 0., False)]
            }
        )
        transprobfactory.add_state_transitions(
            np.array([4, 1], dtype=np.int64),
            {
                'up': [NextStateTuple(np.array([4, 1], dtype=np.int64), 1., 0., False)],
                'down': [NextStateTuple(np.array([5, 1], dtype=np.int64), 1., 0., False)],
                'left': [NextStateTuple(np.array([4, 1], dtype=np.int64), 1., 0., False)],
                'right': [NextStateTuple(np.array([4, 1], dtype=np.int64), 1., 0., False)]
            }
        )
        transprobfactory.add_state_transitions(
            np.array([4, 2], dtype=np.int64),
            {
                'up': [NextStateTuple(np.array([3, 2], dtype=np.int64), 1., 0., False)],
                'down': [NextStateTuple(np.array([4, 2], dtype=np.int64), 1., 0., False)],
                'left': [NextStateTuple(np.array([4, 2], dtype=np.int64), 1., 0., False)],
                'right': [NextStateTuple(np.array([4, 3], dtype=np.int64), 1., 0., False)]
            }
        )
        transprobfactory.add_state_transitions(
            np.array([4, 3], dtype=np.int64),
            {
                'up': [NextStateTuple(np.array([3, 3], dtype=np.int64), 1., 0., False)],
                'down': [NextStateTuple(np.array([4, 3], dtype=np.int64), 1., 0., False)],
                'left': [NextStateTuple(np.array([4, 2], dtype=np.int64), 1., 0., False)],
                'right': [NextStateTuple(np.array([4, 4], dtype=np.int64), 1., 0., False)]
            }
        )
        transprobfactory.add_state_transitions(
            np.array([4, 4], dtype=np.int64),
            {
                'up': [NextStateTuple(np.array([4, 4], dtype=np.int64), 1., 0., False)],
                'down': [NextStateTuple(np.array([5, 4], dtype=np.int64), 1., 1., True)],
                'left': [NextStateTuple(np.array([4, 3], dtype=np.int64), 1., 0., False)],
                'right': [NextStateTuple(np.array([4, 4], dtype=np.int64), 1., 0., False)]
            }
        )
        transprobfactory.add_state_transitions(
            np.array([5, 0], dtype=np.int64),
            {
                'up': [NextStateTuple(np.array([4, 0], dtype=np.int64), 1., 0., False)],
                'down': [NextStateTuple(np.array([5, 0], dtype=np.int64), 1., 0., False)],
                'left': [NextStateTuple(np.array([5, 0], dtype=np.int64), 1., 0., False)],
                'right': [NextStateTuple(np.array([5, 1], dtype=np.int64), 1., 0., False)]
            }
        )
        transprobfactory.add_state_transitions(
            np.array([5, 1], dtype=np.int64),
            {
                'up': [NextStateTuple(np.array([4, 1], dtype=np.int64), 1., 0., False)],
                'down': [NextStateTuple(np.array([5, 1], dtype=np.int64), 1., 0., False)],
                'left': [NextStateTuple(np.array([5, 0], dtype=np.int64), 1., 0., False)],
                'right': [NextStateTuple(np.array([5, 1], dtype=np.int64), 1., 0., False)]
            }
        )
        transprobfactory.add_state_transitions(
            np.array([5, 2], dtype=np.int64),
            {
                'up': [NextStateTuple(np.array([5, 2], dtype=np.int64), 1., 0., False)],
                'down': [NextStateTuple(np.array([5, 2], dtype=np.int64), 1., 0., False)],
                'left': [NextStateTuple(np.array([5, 2], dtype=np.int64), 1., 0., False)],
                'right': [NextStateTuple(np.array([5, 3], dtype=np.int64), 1., 0., False)]
            }
        )
        transprobfactory.add_state_transitions(
            np.array([5, 3], dtype=np.int64),
            {
                'up': [NextStateTuple(np.array([5, 3], dtype=np.int64), 1., 0., False)],
                'down': [NextStateTuple(np.array([5, 3], dtype=np.int64), 1., 0., False)],
                'left': [NextStateTuple(np.array([5, 2], dtype=np.int64), 1., 0., False)],
                'right': [NextStateTuple(np.array([5, 4], dtype=np.int64), 1., 1., True)]
            }
        )
        transprobfactory.add_state_transitions(
            np.array([5, 4], dtype=np.int64),
            {
                'up': [NextStateTuple(np.array([4, 4], dtype=np.int64), 1., 0., False)],
                'down': [NextStateTuple(np.array([5, 4], dtype=np.int64), 1., 1., True)],
                'left': [NextStateTuple(np.array([5, 3], dtype=np.int64), 1., 0., False)],
                'right': [NextStateTuple(np.array([5, 4], dtype=np.int64), 1., 1., True)]
            }
        )

        self.transprobfactory = transprobfactory
        self.maze_state = maze_state
        self.maze_state.set_terminal_given_coordinates([5, 4], True)

    def test_terminal(self):
        print(self.maze_state._terminal_dict)
        for i, j in product(
                range(self.maze_state.x_lowlim, self.maze_state.x_hilim + 1),
                range(self.maze_state.y_lowlim, self.maze_state.y_hilim + 1)
        ):
            self.maze_state.set_state_value([i, j])
            print(f"i: {i}, j: {j}; encoded: {[i, j]}; terminal: {self.maze_state.is_terminal}")
            if (i == self.maze_state.x_hilim) and (j == self.maze_state.y_hilim):
                assert self.maze_state.is_terminal
                assert self.maze_state.get_whether_terminal_given_coordinates([i, j])
            else:
                assert not self.maze_state.is_terminal
                assert not self.maze_state.get_whether_terminal_given_coordinates([i, j])

    def test_policy_iteration(self):
        policy_finder = OptimalPolicyOnValueFunctions(0.85, self.transprobfactory)
        values_dict, policy = policy_finder.policy_iteration()

        for state_value, value in values_dict.items():
            [x, y] = self.maze_state.decode_coordinates(state_value)
            print(f'({x}, {y}): {value}')

        state, actions_dict, _ = self.transprobfactory.generate_mdp_objects()
        assert isinstance(state, DiscreteCategoricalState)

        arrived_destination = False
        for _ in range(state.state_space_size*2):
            action_value = policy.get_action_value(state.state_value)
            print(f'Action value: {action_value}')
            action = policy.get_action(state)
            state = action(state)

            coordinates = self.maze_state.decode_coordinates(state.state_value)
            print(f'at: {coordinates[0]}, {coordinates[1]}')
            if coordinates[0] == 5 and coordinates[1] == 4:
                arrived_destination = True
                break

        assert arrived_destination

    def test_value_iteration(self):
        policy_finder = OptimalPolicyOnValueFunctions(0.85, self.transprobfactory)
        values_dict, policy = policy_finder.value_iteration()

        for state_value, value in values_dict.items():
            [x, y] = self.maze_state.decode_coordinates(state_value)
            print(f'({x}, {y}): {value}')

        state, actions_dict, _ = self.transprobfactory.generate_mdp_objects()
        assert isinstance(state, DiscreteCategoricalState)

        arrived_destination = False
        for _ in range(state.state_space_size*2):
            action_value = policy.get_action_value(state.state_value)
            print(f'Action value: {action_value}')
            action = policy.get_action(state)
            state = action(state)

            coordinates = self.maze_state.decode_coordinates(state.state_value)
            print(f'at: {coordinates[0]}, {coordinates[1]}')
            if coordinates[0] == 5 and coordinates[1] == 4:
                arrived_destination = True
                break

        assert arrived_destination



if __name__ == '__main__':
    unittest.main()
