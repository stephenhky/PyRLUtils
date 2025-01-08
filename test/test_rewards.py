
import unittest

from pyrlutils.openai.utils import OpenAIGymDiscreteEnvironmentTransitionProbabilityFactory


class TestRewards(unittest.TestCase):
    def test_reward1(self):
        tranprobfactory = OpenAIGymDiscreteEnvironmentTransitionProbabilityFactory('FrozenLake-v1')
        state, actions_dict, ind_reward_fcn = tranprobfactory.generate_mdp_objects()


if __name__ == '__main__':
    unittest.main()
