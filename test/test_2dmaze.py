
import unittest

from pyrlutils.transition import TransitionProbabilityFactory, NextStateTuple
from pyrlutils.valuefcns import OptimalPolicyOnValueFunctions
from pyrlutils.state import Discrete2DCartesianState


class Test2DMaze(unittest.TestCase):
    def setUp(self):
        maze_state = Discrete2DCartesianState(0, 5, 0, 4, initial_coordinate=[0, 0])

        transprobfactory = TransitionProbabilityFactory()
        transprobfactory.add_state_transitions(
            maze_state.encode_coordinates([0, 0]),
            {
                'up': [NextStateTuple(maze_state.encode_coordinates([0, 0]), 1., 0., False)],
                'down': [NextStateTuple(maze_state.encode_coordinates([0, 0]), 1., 0., False)],
                'left': [NextStateTuple(maze_state.encode_coordinates([0, 0]), 1., 0., False)],
                'right': [NextStateTuple(maze_state.encode_coordinates([0, 1]), 1., 0., False)]
            }
        )
        transprobfactory.add_state_transitions(
            maze_state.encode_coordinates([0, 1]),
            {
                'up': [NextStateTuple(maze_state.encode_coordinates([0, 1]), 1., 0., False)],
                'down': [NextStateTuple(maze_state.encode_coordinates([1, 1]), 1., 0., False)],
                'left': [NextStateTuple(maze_state.encode_coordinates([0, 0]), 1., 0., False)],
                'right': [NextStateTuple(maze_state.encode_coordinates([0, 2]), 1., 0., False)]
            }
        )
        transprobfactory.add_state_transitions(
            maze_state.encode_coordinates([0, 2]),
            {
                'up': [NextStateTuple(maze_state.encode_coordinates([0, 2]), 1., 0., False)],
                'down': [NextStateTuple(maze_state.encode_coordinates([0, 2]), 1., 0., False)],
                'left': [NextStateTuple(maze_state.encode_coordinates([0, 1]), 1., 0., False)],
                'right': [NextStateTuple(maze_state.encode_coordinates([0, 3]), 1., 0., False)]
            }
        )
        transprobfactory.add_state_transitions(
            maze_state.encode_coordinates([0, 3]),
            {
                'up': [NextStateTuple(maze_state.encode_coordinates([0, 3]), 1., 0., False)],
                'down': [NextStateTuple(maze_state.encode_coordinates([1, 3]), 1., 0., False)],
                'left': [NextStateTuple(maze_state.encode_coordinates([0, 2]), 1., 0., False)],
                'right': [NextStateTuple(maze_state.encode_coordinates([0, 4]), 1., 0., False)]
            }
        )
        transprobfactory.add_state_transitions(
            maze_state.encode_coordinates([0, 4]),
            {
                'up': [NextStateTuple(maze_state.encode_coordinates([0, 4]), 1., 0., False)],
                'down': [NextStateTuple(maze_state.encode_coordinates([1, 4]), 1., 0., False)],
                'left': [NextStateTuple(maze_state.encode_coordinates([0, 3]), 1., 0., False)],
                'right': [NextStateTuple(maze_state.encode_coordinates([0, 4]), 1., 0., False)]
            }
        )
        transprobfactory.add_state_transitions(
            maze_state.encode_coordinates([1, 0]),
            {
                'up': [NextStateTuple(maze_state.encode_coordinates([1, 0]), 1., 0., False)],
                'down': [NextStateTuple(maze_state.encode_coordinates([2, 0]), 1., 0., False)],
                'left': [NextStateTuple(maze_state.encode_coordinates([1, 0]), 1., 0., False)],
                'right': [NextStateTuple(maze_state.encode_coordinates([1, 1]), 1., 0., False)]
            }
        )
        transprobfactory.add_state_transitions(
            maze_state.encode_coordinates([1, 1]),
            {
                'up': [NextStateTuple(maze_state.encode_coordinates([0, 1]), 1., 0., False)],
                'down': [NextStateTuple(maze_state.encode_coordinates([1, 1]), 1., 0., False)],
                'left': [NextStateTuple(maze_state.encode_coordinates([1, 0]), 1., 0., False)],
                'right': [NextStateTuple(maze_state.encode_coordinates([1, 1]), 1., 0., False)]
            }
        )
        transprobfactory.add_state_transitions(
            maze_state.encode_coordinates([1, 2]),
            {
                'up': [NextStateTuple(maze_state.encode_coordinates([1, 2]), 1., 0., False)],
                'down': [NextStateTuple(maze_state.encode_coordinates([2, 2]), 1., 0., False)],
                'left': [NextStateTuple(maze_state.encode_coordinates([1, 2]), 1., 0., False)],
                'right': [NextStateTuple(maze_state.encode_coordinates([1, 3]), 1., 0., False)]
            }
        )
        transprobfactory.add_state_transitions(
            maze_state.encode_coordinates([1, 3]),
            {
                'up': [NextStateTuple(maze_state.encode_coordinates([0, 3]), 1., 0., False)],
                'down': [NextStateTuple(maze_state.encode_coordinates([1, 3]), 1., 0., False)],
                'left': [NextStateTuple(maze_state.encode_coordinates([1, 2]), 1., 0., False)],
                'right': [NextStateTuple(maze_state.encode_coordinates([1, 4]), 1., 0., False)]
            }
        )
        transprobfactory.add_state_transitions(
            maze_state.encode_coordinates([1, 4]),
            {
                'up': [NextStateTuple(maze_state.encode_coordinates([0, 4]), 1., 0., False)],
                'down': [NextStateTuple(maze_state.encode_coordinates([1, 4]), 1., 0., False)],
                'left': [NextStateTuple(maze_state.encode_coordinates([1, 4]), 1., 0., False)],
                'right': [NextStateTuple(maze_state.encode_coordinates([1, 4]), 1., 0., False)]
            }
        )
        transprobfactory.add_state_transitions(
            maze_state.encode_coordinates([2, 0]),
            {
                'up': [NextStateTuple(maze_state.encode_coordinates([1, 0]), 1., 0., False)],
                'down': [NextStateTuple(maze_state.encode_coordinates([3, 0]), 1., 0., False)],
                'left': [NextStateTuple(maze_state.encode_coordinates([2, 0]), 1., 0., False)],
                'right': [NextStateTuple(maze_state.encode_coordinates([2, 1]), 1., 0., False)]
            }
        )
        transprobfactory.add_state_transitions(
            maze_state.encode_coordinates([2, 1]),
            {
                'up': [NextStateTuple(maze_state.encode_coordinates([2, 1]), 1., 0., False)],
                'down': [NextStateTuple(maze_state.encode_coordinates([2, 1]), 1., 0., False)],
                'left': [NextStateTuple(maze_state.encode_coordinates([2, 0]), 1., 0., False)],
                'right': [NextStateTuple(maze_state.encode_coordinates([2, 2]), 1., 0., False)]
            }
        )
        transprobfactory.add_state_transitions(
            maze_state.encode_coordinates([2, 2]),
            {
                'up': [NextStateTuple(maze_state.encode_coordinates([1, 2]), 1., 0., False)],
                'down': [NextStateTuple(maze_state.encode_coordinates([2, 2]), 1., 0., False)],
                'left': [NextStateTuple(maze_state.encode_coordinates([2, 1]), 1., 0., False)],
                'right': [NextStateTuple(maze_state.encode_coordinates([2, 3]), 1., 0., False)]
            }
        )
        transprobfactory.add_state_transitions(
            maze_state.encode_coordinates([2, 3]),
            {
                'up': [NextStateTuple(maze_state.encode_coordinates([2, 3]), 1., 0., False)],
                'down': [NextStateTuple(maze_state.encode_coordinates([3, 3]), 1., 0., False)],
                'left': [NextStateTuple(maze_state.encode_coordinates([2, 2]), 1., 0., False)],
                'right': [NextStateTuple(maze_state.encode_coordinates([2, 3]), 1., 0., False)]
            }
        )
        transprobfactory.add_state_transitions(
            maze_state.encode_coordinates([2, 4]),
            {
                'up': [NextStateTuple(maze_state.encode_coordinates([2, 4]), 1., 0., False)],
                'down': [NextStateTuple(maze_state.encode_coordinates([3, 4]), 1., 0., False)],
                'left': [NextStateTuple(maze_state.encode_coordinates([2, 4]), 1., 0., False)],
                'right': [NextStateTuple(maze_state.encode_coordinates([2, 4]), 1., 0., False)]
            }
        )
        transprobfactory.add_state_transitions(
            maze_state.encode_coordinates([3, 0]),
            {
                'up': [NextStateTuple(maze_state.encode_coordinates([2, 0]), 1., 0., False)],
                'down': [NextStateTuple(maze_state.encode_coordinates([4, 0]), 1., 0., False)],
                'left': [NextStateTuple(maze_state.encode_coordinates([3, 0]), 1., 0., False)],
                'right': [NextStateTuple(maze_state.encode_coordinates([3, 1]), 1., 0., False)]
            }
        )
        transprobfactory.add_state_transitions(
            maze_state.encode_coordinates([3, 1]),
            {
                'up': [NextStateTuple(maze_state.encode_coordinates([3, 1]), 1., 0., False)],
                'down': [NextStateTuple(maze_state.encode_coordinates([3, 1]), 1., 0., False)],
                'left': [NextStateTuple(maze_state.encode_coordinates([3, 0]), 1., 0., False)],
                'right': [NextStateTuple(maze_state.encode_coordinates([3, 2]), 1., 0., False)]
            }
        )
        transprobfactory.add_state_transitions(
            maze_state.encode_coordinates([3, 2]),
            {
                'up': [NextStateTuple(maze_state.encode_coordinates([3, 2]), 1., 0., False)],
                'down': [NextStateTuple(maze_state.encode_coordinates([4, 2]), 1., 0., False)],
                'left': [NextStateTuple(maze_state.encode_coordinates([3, 1]), 1., 0., False)],
                'right': [NextStateTuple(maze_state.encode_coordinates([3, 3]), 1., 0., False)]
            }
        )
        transprobfactory.add_state_transitions(
            maze_state.encode_coordinates([3, 3]),
            {
                'up': [NextStateTuple(maze_state.encode_coordinates([2, 3]), 1., 0., False)],
                'down': [NextStateTuple(maze_state.encode_coordinates([4, 3]), 1., 0., False)],
                'left': [NextStateTuple(maze_state.encode_coordinates([3, 3]), 1., 0., False)],
                'right': [NextStateTuple(maze_state.encode_coordinates([3, 4]), 1., 0., False)]
            }
        )
        transprobfactory.add_state_transitions(
            maze_state.encode_coordinates([3, 4]),
            {
                'up': [NextStateTuple(maze_state.encode_coordinates([2, 4]), 1., 0., False)],
                'down': [NextStateTuple(maze_state.encode_coordinates([3, 4]), 1., 0., False)],
                'left': [NextStateTuple(maze_state.encode_coordinates([3, 3]), 1., 0., False)],
                'right': [NextStateTuple(maze_state.encode_coordinates([3, 4]), 1., 0., False)]
            }
        )
        transprobfactory.add_state_transitions(
            maze_state.encode_coordinates([4, 0]),
            {
                'up': [NextStateTuple(maze_state.encode_coordinates([3, 0]), 1., 0., False)],
                'down': [NextStateTuple(maze_state.encode_coordinates([5, 0]), 1., 0., False)],
                'left': [NextStateTuple(maze_state.encode_coordinates([4, 0]), 1., 0., False)],
                'right': [NextStateTuple(maze_state.encode_coordinates([4, 0]), 1., 0., False)]
            }
        )
        transprobfactory.add_state_transitions(
            maze_state.encode_coordinates([4, 1]),
            {
                'up': [NextStateTuple(maze_state.encode_coordinates([4, 1]), 1., 0., False)],
                'down': [NextStateTuple(maze_state.encode_coordinates([5, 1]), 1., 0., False)],
                'left': [NextStateTuple(maze_state.encode_coordinates([4, 1]), 1., 0., False)],
                'right': [NextStateTuple(maze_state.encode_coordinates([4, 1]), 1., 0., False)]
            }
        )
        transprobfactory.add_state_transitions(
            maze_state.encode_coordinates([4, 2]),
            {
                'up': [NextStateTuple(maze_state.encode_coordinates([3, 2]), 1., 0., False)],
                'down': [NextStateTuple(maze_state.encode_coordinates([4, 2]), 1., 0., False)],
                'left': [NextStateTuple(maze_state.encode_coordinates([4, 2]), 1., 0., False)],
                'right': [NextStateTuple(maze_state.encode_coordinates([4, 3]), 1., 0., False)]
            }
        )
        transprobfactory.add_state_transitions(
            maze_state.encode_coordinates([4, 3]),
            {
                'up': [NextStateTuple(maze_state.encode_coordinates([3, 3]), 1., 0., False)],
                'down': [NextStateTuple(maze_state.encode_coordinates([4, 3]), 1., 0., False)],
                'left': [NextStateTuple(maze_state.encode_coordinates([4, 2]), 1., 0., False)],
                'right': [NextStateTuple(maze_state.encode_coordinates([4, 4]), 1., 0., False)]
            }
        )
        transprobfactory.add_state_transitions(
            maze_state.encode_coordinates([4, 4]),
            {
                'up': [NextStateTuple(maze_state.encode_coordinates([4, 4]), 1., 0., False)],
                'down': [NextStateTuple(maze_state.encode_coordinates([5, 4]), 1., 1., True)],
                'left': [NextStateTuple(maze_state.encode_coordinates([4, 3]), 1., 0., False)],
                'right': [NextStateTuple(maze_state.encode_coordinates([4, 4]), 1., 0., False)]
            }
        )
        transprobfactory.add_state_transitions(
            maze_state.encode_coordinates([5, 0]),
            {
                'up': [NextStateTuple(maze_state.encode_coordinates([4, 0]), 1., 0., False)],
                'down': [NextStateTuple(maze_state.encode_coordinates([5, 0]), 1., 0., False)],
                'left': [NextStateTuple(maze_state.encode_coordinates([5, 0]), 1., 0., False)],
                'right': [NextStateTuple(maze_state.encode_coordinates([5, 1]), 1., 0., False)]
            }
        )
        transprobfactory.add_state_transitions(
            maze_state.encode_coordinates([5, 1]),
            {
                'up': [NextStateTuple(maze_state.encode_coordinates([4, 1]), 1., 0., False)],
                'down': [NextStateTuple(maze_state.encode_coordinates([5, 1]), 1., 0., False)],
                'left': [NextStateTuple(maze_state.encode_coordinates([5, 0]), 1., 0., False)],
                'right': [NextStateTuple(maze_state.encode_coordinates([5, 1]), 1., 0., False)]
            }
        )
        transprobfactory.add_state_transitions(
            maze_state.encode_coordinates([5, 2]),
            {
                'up': [NextStateTuple(maze_state.encode_coordinates([5, 2]), 1., 0., False)],
                'down': [NextStateTuple(maze_state.encode_coordinates([5, 2]), 1., 0., False)],
                'left': [NextStateTuple(maze_state.encode_coordinates([5, 2]), 1., 0., False)],
                'right': [NextStateTuple(maze_state.encode_coordinates([5, 3]), 1., 0., False)]
            }
        )
        transprobfactory.add_state_transitions(
            maze_state.encode_coordinates([5, 3]),
            {
                'up': [NextStateTuple(maze_state.encode_coordinates([5, 3]), 1., 0., False)],
                'down': [NextStateTuple(maze_state.encode_coordinates([5, 3]), 1., 0., False)],
                'left': [NextStateTuple(maze_state.encode_coordinates([5, 2]), 1., 0., False)],
                'right': [NextStateTuple(maze_state.encode_coordinates([5, 4]), 1., 1., True)]
            }
        )
        transprobfactory.add_state_transitions(
            maze_state.encode_coordinates([5, 4]),
            {
                'up': [NextStateTuple(maze_state.encode_coordinates([4, 4]), 1., 0., False)],
                'down': [NextStateTuple(maze_state.encode_coordinates([5, 4]), 1., 1., True)],
                'left': [NextStateTuple(maze_state.encode_coordinates([5, 3]), 1., 0., False)],
                'right': [NextStateTuple(maze_state.encode_coordinates([5, 4]), 1., 1., True)]
            }
        )

        self.transprobfactory = transprobfactory
        self.maze_state = maze_state

    def test_policy_iteration(self):
        policy_finder = OptimalPolicyOnValueFunctions(0.85, self.transprobfactory)
        values_dict, policy = policy_finder.policy_iteration()

        for state_value, value in values_dict.items():
            [x, y] = self.maze_state.decode_coordinates(state_value)
            print('({}, {}): {}'.format(x, y, value))

        state, actions_dict, _ = self.transprobfactory.generate_mdp_objects()

        arrived_destination = False
        for _ in range(state.state_space_size*2):
            action_value = policy.get_action_value(state)
            print('Action value: {}'.format(action_value))
            action = policy.get_action(state)
            state = action(state)

            coordinates = self.maze_state.decode_coordinates(state.state_value)
            print('at: {}, {}'.format(coordinates[0], coordinates[1]))
            if coordinates[0] == 5 and coordinates[1] == 4:
                arrived_destination = True
                break

        assert arrived_destination

    def test_value_iteration(self):
        policy_finder = OptimalPolicyOnValueFunctions(0.85, self.transprobfactory)
        values_dict, policy = policy_finder.value_iteration()

        for state_value, value in values_dict.items():
            [x, y] = self.maze_state.decode_coordinates(state_value)
            print('({}, {}): {}'.format(x, y, value))

        state, actions_dict, _ = self.transprobfactory.generate_mdp_objects()

        arrived_destination = False
        for _ in range(state.state_space_size*2):
            action_value = policy.get_action_value(state)
            print('Action value: {}'.format(action_value))
            action = policy.get_action(state)
            state = action(state)

            coordinates = self.maze_state.decode_coordinates(state.state_value)
            print('at: {}, {}'.format(coordinates[0], coordinates[1]))
            if coordinates[0] == 5 and coordinates[1] == 4:
                arrived_destination = True
                break

        assert arrived_destination



if __name__ == '__main__':
    unittest.main()
