
import unittest

import numpy as np

from pyrlutils.td.utils import decay_schedule
from pyrlutils.td.td import SingleStepTemporalDifferenceLearner, MultipleStepTemporalDifferenceLearner
from pyrlutils.openai.utils import OpenAIGymDiscreteEnvironmentTransitionProbabilityFactory
from pyrlutils.policy import DiscreteDeterminsticPolicy


# the temporal difference learner ensures it runs, not correct values (which cannot be determined)

class TestTD(unittest.TestCase):
    def setUp(self):
        self.transprobfac = OpenAIGymDiscreteEnvironmentTransitionProbabilityFactory('FrozenLake-v1')
        state, actions_dict, rewardfcn = self.transprobfac.generate_mdp_objects()
        self.policy = DiscreteDeterminsticPolicy(actions_dict)
        for i in state.get_all_possible_state_values():
            self.policy.add_deterministic_rule(i, i % 4)

    def test_decay_schedule(self):
        alphas = decay_schedule(1., 0.2, 0.8, 100)
        self.assertAlmostEqual(alphas[0], 1.)
        self.assertAlmostEqual(alphas[80], 0.2)
        np.testing.assert_almost_equal(alphas[80:], np.repeat(0.2, 20))

    def test_singlestep_td_learn(self):
        tdlearner = SingleStepTemporalDifferenceLearner(self.transprobfac, policy=self.policy)
        V, V_track = tdlearner.learn(5)

        assert V.ndim == 1
        assert V.shape[0] == 16

        assert V_track.ndim == 2
        assert V_track.shape[0] == 5
        assert V_track.shape[1] == 16

    def test_multiplestep_td_learn(self):
        tdlearner = MultipleStepTemporalDifferenceLearner(self.transprobfac, policy=self.policy)
        V, V_track = tdlearner.learn(5)

        assert V.ndim == 1
        assert V.shape[0] == 16

        assert V_track.ndim == 2
        assert V_track.shape[0] == 5
        assert V_track.shape[1] == 16


if __name__ == '__main__':
    unittest.main()
