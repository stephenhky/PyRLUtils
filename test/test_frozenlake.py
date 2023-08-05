
import unittest

from pyrlutils.transition import OpenAIGymDiscreteEnvironmentTransitionProbabilityFactory

class TestFrozenLake(unittest.TestCase):
    def test_factory(self):
        tranprobfactory = OpenAIGymDiscreteEnvironmentTransitionProbabilityFactory('FrozenLake-v1')
        state, actions_dict, ind_reward_fcn = tranprobfactory.generate_mdp_objects()

        assert len(state.get_all_possible_state_values()) == 16
        assert state.state_value == 0

        actions_dict[0](state)
        assert state.state_value in {0, 4}

        state.state_value = 15
        actions_dict[2](state)
        assert state.state_value == 15


if __name__ == '__main__':
    unittest.main()
