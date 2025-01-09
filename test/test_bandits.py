
import unittest
from enum import Enum
import random

import numpy as np

from pyrlutils.bandit.reward import IndividualBanditRewardFunction
from pyrlutils.bandit.algo import SimpleBandit, GradientBandit


class BanditWalk(Enum):
    LEFT = 0
    RIGHT = 1


class BanditWalkReward(IndividualBanditRewardFunction):
    def reward(self, action_value: BanditWalk) -> float:
        return 0. if action_value == BanditWalk.LEFT else 1.


class BanditSlipperyWalkReward(IndividualBanditRewardFunction):
    def reward(self, action_value: BanditWalk) -> float:
        r = random.uniform(0, 1)
        if action_value == BanditWalk.LEFT:
            return 0. if r <= 0.8 else 1.
        else:
            return 0. if r <= 0.2 else 1.


class TestBandits(unittest.TestCase):
    def test_simple_bandit_BW(self):
        simple_bandit_BW = SimpleBandit(list(BanditWalk), BanditWalkReward())

        assert simple_bandit_BW._Q.shape[0] == len(list(BanditWalk))
        assert len(simple_bandit_BW.action_values) == len(list(BanditWalk))

        # go for 100 loops
        simple_bandit_BW.loop(100)

        assert simple_bandit_BW.get_action() == BanditWalk.RIGHT

    def test_simple_bandit_BSW(self):
        simple_bandit_BSW = SimpleBandit(list(BanditWalk), BanditSlipperyWalkReward())

        assert simple_bandit_BSW._Q.shape[0] == len(list(BanditWalk))
        assert len(simple_bandit_BSW.action_values) == len(list(BanditWalk))

        # go for 100 loops
        simple_bandit_BSW.loop(100)

        assert simple_bandit_BSW.get_action() == BanditWalk.RIGHT

    def test_gradient_bandit_BW(self):
        gradient_bandit_BW = GradientBandit(list(BanditWalk), BanditWalkReward())

        assert gradient_bandit_BW._preferences.shape[0] == len(list(BanditWalk))
        probs = gradient_bandit_BW._get_probs()
        self.assertAlmostEqual(probs[0], 0.5)
        self.assertAlmostEqual(probs[1], 0.5)

        # go for 100 loops
        gradient_bandit_BW.loop(100)

        assert gradient_bandit_BW.get_action() == BanditWalk.RIGHT

    def test_gradient_bandit_BSW(self):
        gradient_bandit_BSW = GradientBandit(list(BanditWalk), BanditSlipperyWalkReward())

        assert gradient_bandit_BSW._preferences.shape[0] == len(list(BanditWalk))
        probs = gradient_bandit_BSW._get_probs()
        self.assertAlmostEqual(probs[0], 0.5)
        self.assertAlmostEqual(probs[1], 0.5)

        # go for 100 loops
        gradient_bandit_BSW.loop(100)

        assert gradient_bandit_BSW.get_action() == BanditWalk.RIGHT


if __name__ == '__main__':
    unittest.main()
