import unittest

import numpy as np

from recommendation_data_toolbox.lottery import (
    DecisionHistory,
    Lottery,
    LotteryPair,
    unpack_lottery_distribution,
)


class TestLottery(unittest.TestCase):
    def test_lottery(self):
        lottery = Lottery(np.array([3, 5, 2]), np.array([0.1, 0.5, 0.4]))
        self.assertTrue(np.array_equal(lottery.outcomes, np.array([5, 3, 2])))
        self.assertTrue(
            np.array_equal(lottery.probs, np.array([0.5, 0.1, 0.4]))
        )

    def test_decisionHistory(self):
        lot_pair_0 = LotteryPair(
            Lottery(np.array([5, 3, 2]), np.array([0.1, 0.5, 0.4])),
            Lottery(np.array([6, 4, 1]), np.array([0.2, 0.7, 0.1])),
        )
        lot_pair_1 = LotteryPair(
            Lottery(np.array([4, 3, 2]), np.array([0.3, 0.2, 0.5])),
            Lottery(np.array([7, 3, 2]), np.array([0.2, 0.6, 0.2])),
        )
        lottery_pairs = [lot_pair_0, lot_pair_1]
        decision_his = DecisionHistory(lottery_pairs, [True, False])
        idx_0 = LotteryPair(
            Lottery(np.array([5, 3, 2]), np.array([0.1, 0.5, 0.4])),
            Lottery(np.array([6, 4, 1]), np.array([0.2, 0.7, 0.1])),
        )
        idx_1 = LotteryPair(
            Lottery(np.array([4, 3, 2]), np.array([0.3, 0.2, 0.5])),
            Lottery(np.array([7, 3, 2]), np.array([0.2, 0.6, 0.2])),
        )
        self.assertEqual(decision_his[idx_0], True)
        self.assertEqual(decision_his[idx_1], False)

    def test_getLotteryDist_lotNumberEqualsOne(self):
        actual_outcomes, actual_probs = unpack_lottery_distribution(
            20, 0.05, 10, 1, "-"
        )
        expected_outcomes, expected_probs = [20, 10], [0.05, 0.95]
        self.assertTrue(np.array_equal(actual_outcomes, expected_outcomes))
        self.assertTrue(np.allclose(actual_probs, expected_probs))

    def test_getLotteryDist_symm(self):
        expected_outcomes = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        expected_probs = [
            0.00390625,
            0.03125,
            0.109375,
            0.21875,
            0.2734375,
            0.21875,
            0.109375,
            0.03125,
            0.00390625,
        ]
        actual_outcomes, actual_probs = unpack_lottery_distribution(
            4, 1, 0, 9, "Symm"
        )
        self.assertTrue(np.array_equal(actual_outcomes, expected_outcomes))
        self.assertTrue(np.allclose(actual_probs, expected_probs))

    def test_getLotteryDist_rskew(self):
        expected_outcomes = [2, 4, 8, 16, 32, 64, 128, 256, 10]
        expected_probs = [
            0.2,
            0.1,
            0.05,
            0.025,
            0.0125,
            0.00625,
            0.003125,
            0.003125,
            0.6,
        ]
        actual_outcomes, actual_probs = unpack_lottery_distribution(
            9, 0.4, 10, 8, "R-skew"
        )
        self.assertTrue(np.array_equal(actual_outcomes, expected_outcomes))
        self.assertTrue(np.allclose(actual_probs, expected_probs))

    def test_getLotteryDist_lskew(self):
        expected_outcomes = [50, 48, 44, 10]
        expected_probs = [0.2, 0.1, 0.1, 0.6]
        actual_outcomes, actual_probs = unpack_lottery_distribution(
            48, 0.4, 10, 3, "L-skew"
        )
        self.assertTrue(np.array_equal(actual_outcomes, expected_outcomes))
        self.assertTrue(np.allclose(actual_probs, expected_probs))


if __name__ == "__main__":
    unittest.main()
