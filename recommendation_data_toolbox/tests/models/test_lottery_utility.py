import unittest
import numpy as np

from recommendation_data_toolbox.models.lottery_utility import (
    get_lottery_utility_model,
)


class TestLotteryUtility(unittest.TestCase):
    outcomes = None
    outcomes_nonneg = None
    probs = None

    @classmethod
    def setUpClass(cls):
        cls.outcomes = np.array([30, 20, -10])
        cls.outcomes_nonneg = np.array([30, 20, 10])
        cls.probs = np.array([0.2, 0.3, 0.5])

    def test_expectedUtility_powerNonnegOutcomeUtility(self):
        actual = get_lottery_utility_model(
            "expected_utility", "power_nonneg"
        ).lottery_utility_func((0.3,), self.outcomes_nonneg, self.probs)
        expected = 2.28940619609
        self.assertTrue(np.allclose(actual, expected))

    def test_expectedUtility_powerOutcomeUtility(self):
        actual = get_lottery_utility_model(
            "expected_utility", "power"
        ).lottery_utility_func((0.4, 1.5, 0.6), self.outcomes, self.probs)
        expected = -1.21185560577
        self.assertTrue(np.allclose(actual, expected))

    def test_prospectTheory(self):
        actual = get_lottery_utility_model(
            "prospect_theory"
        ).lottery_utility_func((0.3, 0.5), self.outcomes_nonneg, self.probs)
        expected = 2.09671782936
        self.assertTrue(np.allclose(actual, expected))
