import unittest
import numpy as np

from recommendation_data_toolbox.models.outcome_utility import (
    power_uf_on_nonneg_outcomes,
    power_uf_on_real_outcomes,
)


class TestOutcomeUtility(unittest.TestCase):
    def test_powerUfOnNonnegOutcomes(self):
        expected = np.array(
            [1, 2.82842712474619, 5.196152422706632], dtype=np.float64
        )
        actual = power_uf_on_nonneg_outcomes((1.5,), np.array([1, 2, 3]))
        self.assertTrue(np.allclose(actual, expected))

    def test_powerUfOnRealOutcomes(self):
        expected = np.array(
            [-8.485281374238570, -1.5, 0, 1, 1.414213562373095],
            dtype=np.float64,
        )
        actual = power_uf_on_real_outcomes(
            (0.5, 1.5, 2.5), np.array([-2, -1, 0, 1, 2])
        )
        self.assertTrue(np.allclose(actual, expected))
