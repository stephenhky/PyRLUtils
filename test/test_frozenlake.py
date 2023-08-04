
import unittest

from pyrlutils.transition import convert_openai_gymenv_to_transprob

class TestFrozenLake(unittest.TestCase):
    def test_factory(self):
        tranprob = convert_openai_gymenv_to_transprob('FrozenLake-v1')
        state, actions_dict, ind_reward_fcn = tranprob.generate_mdp_objects()

        assert len(state.get_all_possible_state_values()) == 16
