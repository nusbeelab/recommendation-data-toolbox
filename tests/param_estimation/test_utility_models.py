import unittest

import numpy as np

from param_estimation.utility_functions import get_utility_simple


class TestUtilityModel(unittest.TestCase):
    def test_simpleUtilityModel(self):
        expected = np.fromiter([x**1.5 for x in [1, 2, 3]], dtype=np.float64)
        actual = get_utility_simple(dict(r=1.5), np.array([1, 2, 3]))
        self.assertTrue(np.array_equal(actual, expected))
