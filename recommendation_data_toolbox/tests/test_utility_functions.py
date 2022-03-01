import unittest

import numpy as np

from recommendation_data_toolbox.utility_functions import (
    crra1param_utility_fn,
    crra3params_utility_fn,
    get_utility_func_wrapper,
)


class TestUtilityFunctions(unittest.TestCase):
    def __is_within_bound(self, x, lower_bound, upper_bound):
        return x > lower_bound and x < upper_bound

    def test_crra1param_utilityFn(self):
        expected = np.array(
            [1, 2.82842712474619, 5.196152422706632], dtype=np.float64
        )
        actual = crra1param_utility_fn((1.5,), np.array([1, 2, 3]))
        self.assertTrue(np.allclose(actual, expected))

    def test_crra1param_randParams(self):
        utility_func_wrapper = get_utility_func_wrapper("crra1param")
        (param,) = utility_func_wrapper.get_random_params()
        (bound,) = utility_func_wrapper.bounds
        self.assertTrue(self.__is_within_bound(param, *bound))

    def test_crra3params_utilityFn(self):
        expected = np.array(
            [-8.485281374238570, -1.5, 0, 1, 1.414213562373095],
            dtype=np.float64,
        )
        actual = crra3params_utility_fn(
            (0.5, 1.5, 2.5), np.array([-2, -1, 0, 1, 2])
        )
        self.assertTrue(np.allclose(actual, expected))

    def test_crra3params_randParams(self):
        utility_func_wrapper = get_utility_func_wrapper("crra3params")
        r, l, t = utility_func_wrapper.get_random_params()
        r_bound, l_bound, t_bound = utility_func_wrapper.bounds
        self.assertTrue(self.__is_within_bound(r, *r_bound))
        self.assertTrue(self.__is_within_bound(l, *l_bound))
        self.assertTrue(self.__is_within_bound(t, *t_bound))
