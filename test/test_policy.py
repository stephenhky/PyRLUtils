
import unittest

from pyrlutils.policy import DiscreteStochasticPolicy
from pyrlutils.transition import TransitionProbabilityFactory, NextStateTuple


class TestPolicy(unittest.TestCase):
    def test_stochastic_policy(self):
        # refer to DRL pp. 41 (slippery walk)
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
        state, actions_dict, ind_reward_fcn = trans_probs_factory.generate_mdp_objects()

        policy = DiscreteStochasticPolicy(actions_dict)
        policy.add_stochastic_rule(0, ['left', 'right'], probs=[0.5, 0.5])
        policy.add_stochastic_rule(1, ['right'], probs=[1.])
        policy.add_stochastic_rule(2, ['left', 'right'], probs=[0.4, 0.6])

        assert policy.is_stochastic

        assert policy.get_action_value(1) == 'right'
        self.assertAlmostEqual(policy.get_probability(1, 'right'), 1.0)
        self.assertAlmostEqual(policy.get_probability(1, 'left'), 0.0)

        assert policy.get_action_value(0) in ['left', 'right']
        assert policy.get_action_value(0) in ['left', 'right']
        assert policy.get_action_value(0) in ['left', 'right']
        self.assertAlmostEqual(policy.get_probability(0, 'right'), 0.5)
        self.assertAlmostEqual(policy.get_probability(0, 'left'), 0.5)

        assert policy.get_action_value(2) in ['left', 'right']
        assert policy.get_action_value(2) in ['left', 'right']
        assert policy.get_action_value(2) in ['left', 'right']
        self.assertAlmostEqual(policy.get_probability(2, 'right'), 0.6)
        self.assertAlmostEqual(policy.get_probability(2, 'left'), 0.4)


if __name__ == '__main__':
    unittest.main()
