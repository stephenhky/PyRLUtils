
import unittest
from enum import Enum

import random

from pyrlutils.bandit.reward import IndividualBanditRewardFunction
from pyrlutils.bandit.algo import SimpleBandit


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

    def test_simple_bandit_BSW(self):
        simple_bandit_BSW = SimpleBandit(list(BanditWalk), BanditSlipperyWalkReward())

        assert simple_bandit_BSW._Q.shape[0] == len(list(BanditWalk))
        assert len(simple_bandit_BSW.action_values) == len(list(BanditWalk))



if __name__ == '__main__':
    unittest.main()
