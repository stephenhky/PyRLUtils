
import unittest

import numpy as np

from pyrlutils.state import ContinuousState


class TestContinuousSystem(unittest.TestCase):
    def test_1d(self):
        state = ContinuousState(1, np.array([-1.5, 1.5]), init_value=0.0)



if __name__ == '__main__':
    unittest.main()
