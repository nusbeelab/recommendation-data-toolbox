import unittest

import numpy as np

from recommendation_data_toolbox.utility_functions import (
    crra_utility_fn_1param,
    crra_utility_fn_3params,
)


class TestUtilityFunctions(unittest.TestCase):
    def test_crraUtilityFn1param(self):
        expected = np.array(
            [1, 2.82842712474619, 5.196152422706632], dtype=np.float64
        )
        actual = crra_utility_fn_1param((1.5,), np.array([1, 2, 3]))
        self.assertTrue(np.allclose(actual, expected))

    def test_crraUtilityFn3params(self):
        expected = np.array(
            [-8.485281374238570, -1.5, 0, 1, 1.414213562373095],
            dtype=np.float64,
        )
        actual = crra_utility_fn_3params(
            (0.5, 1.5, 2.5), np.array([-2, -1, 0, 1, 2])
        )
        self.assertTrue(np.allclose(actual, expected))
