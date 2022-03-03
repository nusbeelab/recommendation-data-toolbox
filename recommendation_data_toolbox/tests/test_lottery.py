import unittest

from recommendation_data_toolbox.lottery import (
    Lottery,
    unpack_lottery_distribution,
)


class TestLottery(unittest.TestCase):
    def test_getLotteryDist_lotNumberEqualsOne(self):
        actual_lottery = unpack_lottery_distribution(20, 0.05, 10, 1, "-")
        expected_lottery = Lottery([20, 10], [0.05, 0.95])
        self.assertEqual(actual_lottery, expected_lottery)

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
        expected_lottery = Lottery(expected_values, expected_probs)
        actual_lottery = unpack_lottery_distribution(4, 1, 0, 9, "Symm")
        self.assertEqual(actual_lottery, expected_lottery)

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
        expected_lottery = Lottery(expected_values, expected_probs)

        actual_lottery = unpack_lottery_distribution(9, 0.4, 10, 8, "R-skew")
        self.assertEqual(actual_lottery, expected_lottery)

    def test_getLotteryDist_lskew(self):
        expected_values = [50, 48, 44, 10]
        expected_probs = [0.2, 0.1, 0.1, 0.6]
        expected_lottery = Lottery(expected_values, expected_probs)

        actual_lottery = unpack_lottery_distribution(48, 0.4, 10, 3, "L-skew")
        self.assertEqual(actual_lottery, expected_lottery)


if __name__ == "__main__":
    unittest.main()
