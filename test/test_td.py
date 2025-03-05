
import unittest

import numpy as np

from pyrlutils.td.utils import decay_schedule


class TestTD(unittest.TestCase):
    def test_decay_schedule(self):
        alphas = decay_schedule(1., 0.2, 0.8, 100)
        self.assertAlmostEqual(alphas[0], 1.)
        self.assertAlmostEqual(alphas[80], 0.2)
        np.testing.assert_almost_equal(alphas[80:], np.repeat(0.2, 20))


if __name__ == '__main__':
    unittest.main()
