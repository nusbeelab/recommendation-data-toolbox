import unittest
import numpy as np

from recommendation_data_toolbox.lottery import Lottery
from recommendation_data_toolbox.models.lottery_utility import (
    get_lottery_utility_model,
)


class TestLotteryUtility(unittest.TestCase):
    lottery = None
    nonneg_outcome_lotery = None

    @classmethod
    def setUpClass(cls):
        cls.lottery = Lottery(
            np.array([30, 20, -10]), np.array([0.2, 0.3, 0.5])
        )
        cls.nonneg_outcome_lotery = Lottery(
            np.array([30, 20, 10]), np.array([0.2, 0.3, 0.5])
        )

    def test_expectedUtility_1paramOutcomeUtility(self):
        actual = get_lottery_utility_model(
            "expected_utility", "1param"
        ).lottery_utility_func((0.3,), self.nonneg_outcome_lotery)
        expected = 2.28940619609
        self.assertTrue(np.allclose(actual, expected))

    def test_expectedUtility_3paramsOutcomeUtility(self):
        actual = get_lottery_utility_model(
            "expected_utility", "3params"
        ).lottery_utility_func((0.4, 1.5, 0.6), self.lottery)
        expected = -1.21185560577
        self.assertTrue(np.allclose(actual, expected))

    def test_prospectTheory(self):
        actual = get_lottery_utility_model(
            "prospect_theory"
        ).lottery_utility_func((0.3, 0.5), self.nonneg_outcome_lotery)
        expected = 2.09671782936
        self.assertTrue(np.allclose(actual, expected))
