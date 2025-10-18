
import unittest

import numpy as np

from pyrlutils.td.sarsa import SARSALearner
from pyrlutils.td.qlearn import QLearner
from pyrlutils.td.utils import decay_schedule
from pyrlutils.td.state_td import SingleStepTemporalDifferenceLearner, MultipleStepTemporalDifferenceLearner
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
        self.action_values = list(actions_dict.keys())

    def test_decay_schedule(self):
        alphas = decay_schedule(1., 0.2, 0.8, 100)
        self.assertAlmostEqual(alphas[0], 1.)
        self.assertAlmostEqual(alphas[80], 0.2)
        np.testing.assert_almost_equal(alphas[80:], np.repeat(0.2, 20))

    def test_singlestep_td_learn(self):
        tdlearner = SingleStepTemporalDifferenceLearner(self.transprobfac, policy=self.policy)
        V, V_track = tdlearner.learn(5)

        assert V.tensor_dimensions == 1
        assert V.dimension_sizes[0] == 16

        assert V_track.tensor_dimensions == 2
        assert V_track.dimension_sizes[0] == 5
        assert V_track.dimension_sizes[1] == 16

    def test_multiplestep_td_learn(self):
        tdlearner = MultipleStepTemporalDifferenceLearner(self.transprobfac, policy=self.policy)
        V, V_track = tdlearner.learn(5)

        assert V.tensor_dimensions == 1
        assert V.dimension_sizes[0] == 16

        assert V_track.tensor_dimensions == 2
        assert V_track.dimension_sizes[0] == 5
        assert V_track.dimension_sizes[1] == 16

    def test_sarsa_learn(self):
        sarsalearner = SARSALearner(self.transprobfac)
        Q, V, pi, Q_track, pi_track = sarsalearner.learn(200)

        assert Q.tensor_dimensions == 2
        assert Q.dimension_sizes[0] == 16
        assert Q.dimension_sizes[1] == len(self.action_values)

        assert V.tensor_dimensions == 1
        assert V.dimension_sizes[0] == 16

        assert Q_track.tensor_dimensions == 3
        assert Q_track.dimension_sizes[0] == 200
        assert Q_track.dimension_sizes[1] == 16
        assert Q_track.dimension_sizes[2] == len(self.action_values)
        assert len(pi_track) == 200

    def test_q_learn(self):
        qlearner = QLearner(self.transprobfac)
        Q, V, pi, Q_track, pi_track = qlearner.learn(150)

        assert Q.tensor_dimensions == 2
        assert Q.dimension_sizes[0] == 16
        assert Q.dimension_sizes[1] == len(self.action_values)

        assert V.tensor_dimensions == 1
        assert V.dimension_sizes[0] == 16

        assert Q_track.tensor_dimensions == 3
        assert Q_track.dimension_sizes[0] == 150
        assert Q_track.dimension_sizes[1] == 16
        assert Q_track.dimension_sizes[2] == len(self.action_values)
        assert len(pi_track) == 150


if __name__ == '__main__':
    unittest.main()
