
import unittest

from pyrlutils.transition import TransitionProbabilityFactory, NextStateTuple
from pyrlutils.valuefcns import OptimalPolicyOnValueFunctions


class Test2DMaze(unittest.TestCase):
    def setUp(self):
        # maze_state = Discrete2DCartesianState(0, 5, 0, 4, iniial_coordinate=(0, 0))

        transprobfactory = TransitionProbabilityFactory()
        transprobfactory.add_state_transitions(
            (0, 0),
            {
                'up': [NextStateTuple((0, 0), 1., 0., False)],
                'down': [NextStateTuple((0, 0), 1., 0., False)],
                'left': [NextStateTuple((0, 0), 1., 0., False)],
                'right': [NextStateTuple((0, 1), 1., 0., False)]
            }
        )
        transprobfactory.add_state_transitions(
            (0, 1),
            {
                'up': [NextStateTuple((0, 1), 1., 0., False)],
                'down': [NextStateTuple((1, 1), 1., 0., False)],
                'left': [NextStateTuple((0, 0), 1., 0., False)],
                'right': [NextStateTuple((0, 2), 1., 0., False)]
            }
        )
        transprobfactory.add_state_transitions(
            (0, 2),
            {
                'up': [NextStateTuple((0, 2), 1., 0., False)],
                'down': [NextStateTuple((0, 2), 1., 0., False)],
                'left': [NextStateTuple((0, 1), 1., 0., False)],
                'right': [NextStateTuple((0, 3), 1., 0., False)]
            }
        )
        transprobfactory.add_state_transitions(
            (0, 3),
            {
                'up': [NextStateTuple((0, 3), 1., 0., False)],
                'down': [NextStateTuple((1, 3), 1., 0., False)],
                'left': [NextStateTuple((0, 2), 1., 0., False)],
                'right': [NextStateTuple((0, 4), 1., 0., False)]
            }
        )
        transprobfactory.add_state_transitions(
            (0, 4),
            {
                'up': [NextStateTuple((0, 4), 1., 0., False)],
                'down': [NextStateTuple((1, 4), 1., 0., False)],
                'left': [NextStateTuple((0, 3), 1., 0., False)],
                'right': [NextStateTuple((0, 4), 1., 0., False)]
            }
        )
        transprobfactory.add_state_transitions(
            (1, 0),
            {
                'up': [NextStateTuple((1, 0), 1., 0., False)],
                'down': [NextStateTuple((2, 0), 1., 0., False)],
                'left': [NextStateTuple((1, 0), 1., 0., False)],
                'right': [NextStateTuple((1, 1), 1., 0., False)]
            }
        )
        transprobfactory.add_state_transitions(
            (1, 1),
            {
                'up': [NextStateTuple((0, 1), 1., 0., False)],
                'down': [NextStateTuple((1, 1), 1., 0., False)],
                'left': [NextStateTuple((1, 0), 1., 0., False)],
                'right': [NextStateTuple((1, 1), 1., 0., False)]
            }
        )
        transprobfactory.add_state_transitions(
            (1, 2),
            {
                'up': [NextStateTuple((1, 2), 1., 0., False)],
                'down': [NextStateTuple((2, 2), 1., 0., False)],
                'left': [NextStateTuple((1, 2), 1., 0., False)],
                'right': [NextStateTuple((1, 3), 1., 0., False)]
            }
        )
        transprobfactory.add_state_transitions(
            (1, 3),
            {
                'up': [NextStateTuple((0, 3), 1., 0., False)],
                'down': [NextStateTuple((1, 3), 1., 0., False)],
                'left': [NextStateTuple((1, 2), 1., 0., False)],
                'right': [NextStateTuple((1, 4), 1., 0., False)]
            }
        )
        transprobfactory.add_state_transitions(
            (1, 4),
            {
                'up': [NextStateTuple((0, 4), 1., 0., False)],
                'down': [NextStateTuple((1, 4), 1., 0., False)],
                'left': [NextStateTuple((1, 4), 1., 0., False)],
                'right': [NextStateTuple((1, 4), 1., 0., False)]
            }
        )
        transprobfactory.add_state_transitions(
            (2, 0),
            {
                'up': [NextStateTuple((1, 0), 1., 0., False)],
                'down': [NextStateTuple((3, 0), 1., 0., False)],
                'left': [NextStateTuple((2, 0), 1., 0., False)],
                'right': [NextStateTuple((2, 1), 1., 0., False)]
            }
        )
        transprobfactory.add_state_transitions(
            (2, 1),
            {
                'up': [NextStateTuple((2, 1), 1., 0., False)],
                'down': [NextStateTuple((2, 1), 1., 0., False)],
                'left': [NextStateTuple((2, 0), 1., 0., False)],
                'right': [NextStateTuple((2, 2), 1., 0., False)]
            }
        )
        transprobfactory.add_state_transitions(
            (2, 2),
            {
                'up': [NextStateTuple((1, 2), 1., 0., False)],
                'down': [NextStateTuple((2, 2), 1., 0., False)],
                'left': [NextStateTuple((2, 1), 1., 0., False)],
                'right': [NextStateTuple((2, 3), 1., 0., False)]
            }
        )
        transprobfactory.add_state_transitions(
            (2, 3),
            {
                'up': [NextStateTuple((2, 3), 1., 0., False)],
                'down': [NextStateTuple((3, 3), 1., 0., False)],
                'left': [NextStateTuple((2, 2), 1., 0., False)],
                'right': [NextStateTuple((2, 3), 1., 0., False)]
            }
        )
        transprobfactory.add_state_transitions(
            (2, 4),
            {
                'up': [NextStateTuple((2, 4), 1., 0., False)],
                'down': [NextStateTuple((3, 4), 1., 0., False)],
                'left': [NextStateTuple((2, 4), 1., 0., False)],
                'right': [NextStateTuple((2, 4), 1., 0., False)]
            }
        )
        transprobfactory.add_state_transitions(
            (3, 0),
            {
                'up': [NextStateTuple((2, 0), 1., 0., False)],
                'down': [NextStateTuple((4, 0), 1., 0., False)],
                'left': [NextStateTuple((3, 0), 1., 0., False)],
                'right': [NextStateTuple((3, 1), 1., 0., False)]
            }
        )
        transprobfactory.add_state_transitions(
            (3, 1),
            {
                'up': [NextStateTuple((3, 1), 1., 0., False)],
                'down': [NextStateTuple((3, 1), 1., 0., False)],
                'left': [NextStateTuple((3, 0), 1., 0., False)],
                'right': [NextStateTuple((3, 2), 1., 0., False)]
            }
        )
        transprobfactory.add_state_transitions(
            (3, 2),
            {
                'up': [NextStateTuple((3, 2), 1., 0., False)],
                'down': [NextStateTuple((4, 2), 1., 0., False)],
                'left': [NextStateTuple((3, 1), 1., 0., False)],
                'right': [NextStateTuple((3, 3), 1., 0., False)]
            }
        )
        transprobfactory.add_state_transitions(
            (3, 3),
            {
                'up': [NextStateTuple((2, 3), 1., 0., False)],
                'down': [NextStateTuple((4, 3), 1., 0., False)],
                'left': [NextStateTuple((3, 3), 1., 0., False)],
                'right': [NextStateTuple((3, 4), 1., 0., False)]
            }
        )
        transprobfactory.add_state_transitions(
            (3, 4),
            {
                'up': [NextStateTuple((2, 4), 1., 0., False)],
                'down': [NextStateTuple((3, 4), 1., 0., False)],
                'left': [NextStateTuple((3, 3), 1., 0., False)],
                'right': [NextStateTuple((3, 4), 1., 0., False)]
            }
        )
        transprobfactory.add_state_transitions(
            (4, 0),
            {
                'up': [NextStateTuple((3, 0), 1., 0., False)],
                'down': [NextStateTuple((5, 0), 1., 0., False)],
                'left': [NextStateTuple((4, 0), 1., 0., False)],
                'right': [NextStateTuple((4, 0), 1., 0., False)]
            }
        )
        transprobfactory.add_state_transitions(
            (4, 1),
            {
                'up': [NextStateTuple((4, 1), 1., 0., False)],
                'down': [NextStateTuple((5, 1), 1., 0., False)],
                'left': [NextStateTuple((4, 1), 1., 0., False)],
                'right': [NextStateTuple((4, 1), 1., 0., False)]
            }
        )
        transprobfactory.add_state_transitions(
            (4, 2),
            {
                'up': [NextStateTuple((3, 2), 1., 0., False)],
                'down': [NextStateTuple((4, 2), 1., 0., False)],
                'left': [NextStateTuple((4, 2), 1., 0., False)],
                'right': [NextStateTuple((4, 3), 1., 0., False)]
            }
        )
        transprobfactory.add_state_transitions(
            (4, 3),
            {
                'up': [NextStateTuple((3, 3), 1., 0., False)],
                'down': [NextStateTuple((4, 3), 1., 0., False)],
                'left': [NextStateTuple((4, 2), 1., 0., False)],
                'right': [NextStateTuple((4, 4), 1., 0., False)]
            }
        )
        transprobfactory.add_state_transitions(
            (4, 4),
            {
                'up': [NextStateTuple((4, 4), 1., 0., False)],
                'down': [NextStateTuple((5, 4), 1., 0., True)],
                'left': [NextStateTuple((4, 3), 1., 0., False)],
                'right': [NextStateTuple((4, 4), 1., 0., False)]
            }
        )
        transprobfactory.add_state_transitions(
            (5, 0),
            {
                'up': [NextStateTuple((4, 0), 1., 0., False)],
                'down': [NextStateTuple((5, 0), 1., 0., False)],
                'left': [NextStateTuple((5, 0), 1., 0., False)],
                'right': [NextStateTuple((5, 1), 1., 0., False)]
            }
        )
        transprobfactory.add_state_transitions(
            (5, 1),
            {
                'up': [NextStateTuple((4, 1), 1., 0., False)],
                'down': [NextStateTuple((5, 1), 1., 0., False)],
                'left': [NextStateTuple((5, 0), 1., 0., False)],
                'right': [NextStateTuple((5, 1), 1., 0., False)]
            }
        )
        transprobfactory.add_state_transitions(
            (5, 2),
            {
                'up': [NextStateTuple((5, 2), 1., 0., False)],
                'down': [NextStateTuple((5, 2), 1., 0., False)],
                'left': [NextStateTuple((5, 2), 1., 0., False)],
                'right': [NextStateTuple((5, 3), 1., 0., False)]
            }
        )
        transprobfactory.add_state_transitions(
            (5, 3),
            {
                'up': [NextStateTuple((5, 3), 1., 0., False)],
                'down': [NextStateTuple((5, 3), 1., 0., False)],
                'left': [NextStateTuple((5, 2), 1., 0., False)],
                'right': [NextStateTuple((5, 4), 1., 0., True)]
            }
        )
        transprobfactory.add_state_transitions(
            (5, 4),
            {
                'up': [NextStateTuple((4, 4), 1., 0., False)],
                'down': [NextStateTuple((5, 4), 1., 0., True)],
                'left': [NextStateTuple((5, 3), 1., 0., False)],
                'right': [NextStateTuple((5, 4), 1., 0., True)]
            }
        )

        self.transprobfactory = transprobfactory

    def test_policy_iteration(self):
        policy_finder = OptimalPolicyOnValueFunctions(0.85, self.transprobfactory)
        values_dict, policy = policy_finder.policy_iteration()

        for state_value, value in values_dict.items():
            print('{}: {}'.format(state_value, value))

        state, actions_dict, _ = self.transprobfactory.generate_mdp_objects()

        arrived_destination = False
        for _ in range(state.state_space_size*2):
            action = policy.get_action(state)
            action(state)

            if state.state_value[0] == 5 and state.state_value[1] == 4:
                arrived_destination = True
                break

        assert arrived_destination



if __name__ == '__main__':
    unittest.main()
