import unittest
import numpy as np

from recommendation_data_toolbox.mle import eu_fn, neg_log_lik_fn


class TestMle(unittest.TestCase):
    def test_getEu_1d(self):
        actual = eu_fn(
            params=(1,),
            utility_fn=lambda params, x: x + params[0],
            values=np.array([1, 2, 3]),
            probs=np.array([0.4, 0.5, 0.1]),
        )
        self.assertTrue(np.allclose(actual, 2.7))

    def test_getEu_2d(self):
        actual = eu_fn(
            params=(1,),
            utility_fn=lambda params, x: x + params[0],
            values=np.array([[1, 2, 3], [4, 5, 6]]),
            probs=np.array([[0.4, 0.5, 0.1], [0.3, 0.5, 0.2]]),
        )
        self.assertTrue(np.allclose(actual, np.array([2.7, 5.9])))

    def test_getNegLogLik(self):
        actual = neg_log_lik_fn(
            params=(1,),
            utility_fn=lambda params, x: x + params[0],
            a_values=np.array([[1, 2, 3], [4, 5, 6]]),
            a_probs=np.array([[0.4, 0.5, 0.1], [0.3, 0.5, 0.2]]),
            b_values=np.array([[7, 8, 9], [10, 11, 12]]),
            b_probs=np.array([[0.3, 0.5, 0.2], [0.4, 0.5, 0.1]]),
            observed_data=np.array([0, 1]),
        )
        expected = 21.987994803602348
        self.assertTrue(np.allclose(actual, expected))
