import unittest
import numpy as np
import numpy.typing as npt

from recommendation_data_toolbox.mle import neg_log_lik_fn


class TestMle(unittest.TestCase):
    def test_negLogLikFn(self):
        def lottery_utility_func(
            params: tuple,
            ocs: npt.NDArray[np.int_],
            probs: npt.NDArray[np.float64],
        ):
            return np.sum(np.multiply(ocs + params[0], probs), axis=-1)

        actual = neg_log_lik_fn(
            params=(1,),
            lottery_utility_func=lottery_utility_func,
            a_ocs=np.array([[1, 2, 3], [4, 5, 6]]),
            a_probs=np.array([[0.4, 0.5, 0.1], [0.3, 0.5, 0.2]]),
            b_ocs=np.array([[7, 8, 9], [10, 11, 12]]),
            b_probs=np.array([[0.3, 0.5, 0.2], [0.4, 0.5, 0.1]]),
            observed_data=np.array([0, 1]),
        )
        expected = 21.987994803602348
        self.assertTrue(np.allclose(actual, expected))
