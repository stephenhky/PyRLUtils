
import unittest

import numpy as np
from npdict import NumpyNDArrayWrappedDict

from pyrlutils.td.utils import select_action


class TestQSelectionAction(unittest.TestCase):
    def test_numpy(self):
        q_matrix = np.array([[1.2, 3.4, 2.4],
                             [9.7, 0.4, 2.3]])
        assert select_action(0, q_matrix, 0.0) == 1
        assert select_action(1, q_matrix, 0.0) == 0

    def test_npdict(self):
        q_matrix = NumpyNDArrayWrappedDict.from_numpyarray_given_keywords(
            [['cat', 'dog', 'fox'], ['north', 'east', 'south', 'west']],
            np.array([[1.2, 3.3, 2.1, -0.2],
                      [9.7, 0.4, 2.2, 10.9],
                      [0.1, -2.3, -10.9, 0.5]])
        )
        assert select_action('cat', q_matrix, 0.0) == 'east'
        assert select_action('dog', q_matrix, 0.0) == 'west'
        assert select_action('fox', q_matrix, 0.0) == 'west'


if __name__ == '__main__':
    unittest.main()
