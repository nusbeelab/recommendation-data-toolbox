import unittest
import numpy as np

from recommendation_data_toolbox.lottery_utility.outcome_utility import (
    power_uf_on_nonneg_ocs,
    power_uf_on_real_ocs,
)


class TestOutcomeUtility(unittest.TestCase):
    def test_powerUfOnNonnegOcs(self):
        expected = np.array(
            [1, 2.82842712474619, 5.196152422706632], dtype=np.float64
        )
        actual = power_uf_on_nonneg_ocs((1.5,), np.array([1, 2, 3]))
        self.assertTrue(np.allclose(actual, expected))

    def test_powerUfOnRealOcs(self):
        expected = np.array(
            [-8.485281374238570, -1.5, 0, 1, 1.414213562373095],
            dtype=np.float64,
        )
        actual = power_uf_on_real_ocs(
            (0.5, 1.5, 2.5), np.array([-2, -1, 0, 1, 2])
        )
        self.assertTrue(np.allclose(actual, expected))
