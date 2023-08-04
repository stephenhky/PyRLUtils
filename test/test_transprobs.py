
import unittest

from pyrlutils.transition import TransitionProbabilityFactory, NextStateTuple

class TestState(unittest.TestCase):
    def test_slippery_walk(self):
        # refer to DRL pp. 41
        trans_probs_factory = TransitionProbabilityFactory()
        trans_probs_factory.add_state_transitions(
            0,
            {
                'left': [NextStateTuple(0, 1.0, 0.0, True)],
                'right': [NextStateTuple(0, 1.0, 0.0, True)]
            }
        )
        trans_probs_factory.add_state_transitions(
            1,
            {
                'left': [NextStateTuple(0, 0.8, 0.0, True), NextStateTuple(2, 0.2, 1.0, True)],
                'right': [NextStateTuple(2, 0.8, 1.0, True), NextStateTuple(0, 0.2, 0.0, True)]
            }
        )
        trans_probs_factory.add_state_transitions(
            2,
            {
                'left': [NextStateTuple(2, 1.0, 1.0, True)],
                'right': [NextStateTuple(2, 1.0, 1.0, True)]
            }
        )

        state, actions, ind_reward_fcn = trans_probs_factory.generate_mdp_objects()

        assert len(state.get_all_possible_state_values()) == 3
        assert len(actions) == 2

        state = actions['left'](state)
        assert state.state_value == 0
        state = actions['right'](state)
        assert state.state_value == 0

        state.set_state_value(2)
        state = actions['left'](state)
        assert state.state_value == 2
        state = actions['right'](state)
        assert state.state_value == 2

        state.set_state_value(1)
        state = actions['left'](state)
        assert state.state_value != 1
        state.set_state_value(1)
        state = actions['right'](state)
        assert state.state_value != 1

        assert ind_reward_fcn(0, 'left', 0) == 0.0
        assert ind_reward_fcn(0, 'right', 0) == 0.0
        assert ind_reward_fcn(0, 'left', 1) == 0.0
        assert ind_reward_fcn(0, 'right', 1) == 0.0
        assert ind_reward_fcn(0, 'left', 2) == 0.0
        assert ind_reward_fcn(0, 'right', 2) == 0.0
        assert ind_reward_fcn(1, 'left', 0) == 0.0
        assert ind_reward_fcn(1, 'left', 2) == 1.0
        assert ind_reward_fcn(1, 'right', 0) == 0.0
        assert ind_reward_fcn(1, 'right', 2) == 1.0
        assert ind_reward_fcn(2, 'left', 2) == 1.0
        assert ind_reward_fcn(2, 'right', 2) == 1.0


if __name__ == '__main__':
    unittest.main()
