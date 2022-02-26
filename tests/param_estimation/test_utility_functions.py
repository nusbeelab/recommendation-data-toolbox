import unittest

import numpy as np

from param_estimation.utility_functions import utility_fn_1param


class TestUtilityFunctions(unittest.TestCase):
    def test_utilityFn1param(self):
        expected = np.fromiter([x**1.5 for x in [1, 2, 3]], dtype=np.float64)
        actual = utility_fn_1param(dict(r=1.5), np.array([1, 2, 3]))
        self.assertTrue(np.array_equal(actual, expected))
