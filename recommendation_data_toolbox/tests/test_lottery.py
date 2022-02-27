import unittest

import numpy as np

from recommendation_data_toolbox.lottery import unpack_lottery_distribution


class TestLottery(unittest.TestCase):
    def test_getLotteryDist_lotNumberEqualsOne(self):
        actual_values, actual_probs = unpack_lottery_distribution(
            20, 0.05, 10, 1, "-"
        )
        self.assertTrue(np.array_equal(actual_values, np.array([20, 10])))
        self.assertTrue(np.array_equal(actual_probs, np.array([0.05, 0.95])))

    def test_getLotteryDist_symm(self):
        expected_values = [0, 1, 2, 3, 4, 5, 6, 7, 8]
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
        actual_values, actual_probs = unpack_lottery_distribution(
            4, 1, 0, 9, "Symm"
        )
        self.assertTrue(np.array_equal(actual_values, expected_values))
        self.assertTrue(np.array_equal(actual_probs, expected_probs))

    def test_getLotteryDist_rskew(self):
        expected_values = [2, 4, 8, 16, 32, 64, 128, 256, 10]
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

        actual_values, actual_probs = unpack_lottery_distribution(
            9, 0.4, 10, 8, "R-skew"
        )
        self.assertTrue(np.array_equal(actual_values, expected_values))
        self.assertTrue(np.array_equal(actual_probs, expected_probs))

    def test_getLotteryDist_lskew(self):
        expected_values = [50, 48, 44, 10]
        expected_probs = [0.2, 0.1, 0.1, 0.6]

        actual_values, actual_probs = unpack_lottery_distribution(
            48, 0.4, 10, 3, "L-skew"
        )
        self.assertTrue(np.array_equal(actual_values, expected_values))
        self.assertTrue(np.array_equal(actual_probs, expected_probs))


if __name__ == "__main__":
    unittest.main()