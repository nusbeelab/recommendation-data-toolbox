import unittest

import numpy as np
from recommendation_data_toolbox.lottery import Lottery, Problem

from recommendation_data_toolbox.rec.content_based import (
    fosd,
    get_features_per_problem,
    get_lot_features,
)
from recommendation_data_toolbox.tests.mock_data import PROBLEMS


class TestContentBased(unittest.TestCase):
    def test_getFeatures(self):
        lot = Lottery(np.array([5, 3, 2]), np.array([0.1, 0.5, 0.4]))
        actual = get_lot_features(lot)
        expected = np.array(
            [
                5,
                2,
                3,
                3,
                5,
                3,
                2.8,
                0.871779789,
                1.304047328,
                0,
            ]
        )
        self.assertTrue(np.allclose(actual, expected))

    def test_getFeaturesPerProblem(self):
        problem = Problem(
            Lottery(np.array([5, 3, 2]), np.array([0.1, 0.5, 0.4])),
            Lottery(np.array([6, 4, 1]), np.array([0.2, 0.6, 0.2])),
        )
        actual = get_features_per_problem(problem)
        expected = np.array(
            [
                5,
                2,
                3,
                3,
                5,
                3,
                2.8,
                0.87177979,
                1.30404734,
                0,
                6,
                1,
                5,
                4,
                6,
                3,
                3.8,
                1.6,
                -0.55078125,
                0,
                1,
                -1,
                2,
                1,
                1,
                0,
                1,
                0.72822021,
                -1.85482858,
                0,
                1,
                1.35714286,
                0,
                0,
            ]
        )
        self.assertTrue(np.allclose(actual, expected))

    def test_fosd_1(self):
        lot_a = Lottery([0, 105, 175], [0.08, 0.8, 0.12])
        lot_b = Lottery([0, 105, 175], [0.28, 0.6, 0.12])
        self.assertTrue(fosd(lot_a, lot_b))
        self.assertFalse(fosd(lot_b, lot_a))

    def test_fosd_2(self):
        lot_a = Lottery([1, 2], [0.5, 0.5])
        lot_b = Lottery([3, 1], [0.5, 0.5])
        self.assertFalse(fosd(lot_a, lot_b))
        self.assertTrue(fosd(lot_b, lot_a))

    def test_fosd_3(self):
        lot_a = Lottery([1, 2, 3], [0.1, 0.4, 0.5])
        lot_b = Lottery([2, 3, 1], [0.2, 0.6, 0.2])
        self.assertFalse(fosd(lot_a, lot_b))
        self.assertFalse(fosd(lot_b, lot_a))


if __name__ == "__main__":
    unittest.main()
