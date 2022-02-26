import unittest

import numpy as np
from param_estimation.raw_data_transform import get_unnormalized_hb_lottery


class TestRawDataTransform(unittest.TestCase):
    def test_getUnnormHbLot_lotNumberEqualsOne(self):
        actual_values, actual_probs = get_unnormalized_hb_lottery(
            20, 0.05, 1, "-"
        )
        self.assertTrue(np.array_equal(actual_values, np.array([20])))
        self.assertTrue(np.array_equal(actual_probs, np.array([0.05])))

    def test_getUnnormHbLot_symm(self):
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
        actual_values, actual_probs = get_unnormalized_hb_lottery(
            4, 1, 9, "Symm"
        )
        self.assertTrue(np.array_equal(actual_values, expected_values))
        self.assertTrue(np.array_equal(actual_probs, expected_probs))

    def test_getUnnormHbLot_rskew(self):
        expected_values = [2, 4, 8, 16, 32, 64, 128, 256]
        expected_probs = [
            0.5,
            0.25,
            0.125,
            0.0625,
            0.03125,
            0.015625,
            0.0078125,
            0.0078125,
        ]

        actual_values, actual_probs = get_unnormalized_hb_lottery(
            9, 1, 8, "R-skew"
        )
        self.assertTrue(np.array_equal(actual_values, expected_values))
        self.assertTrue(np.array_equal(actual_probs, expected_probs))

    def test_getUnnormHbLot_lskew(self):
        expected_values = [50, 48, 44]
        expected_probs = [0.2, 0.1, 0.1]

        actual_values, actual_probs = get_unnormalized_hb_lottery(
            48, 0.4, 3, "L-skew"
        )
        self.assertTrue(np.array_equal(actual_values, expected_values))
        self.assertTrue(np.array_equal(actual_probs, expected_probs))


if __name__ == "__main__":
    unittest.main()
