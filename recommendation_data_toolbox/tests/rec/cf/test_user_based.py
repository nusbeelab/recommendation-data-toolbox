import unittest

import numpy as np

from recommendation_data_toolbox.rec.cf.user_based import (
    UcbfRecommender,
    same_decision_frac,
)
from recommendation_data_toolbox.lottery import DecisionHistory
from recommendation_data_toolbox.tests.mock_data import (
    lot_pair_0,
    lot_pair_1,
    lot_pair_2,
    lot_pair_3,
    lot_pair_4,
)


class TestUserBased(unittest.TestCase):
    def test_sameDecisionFrac(self):
        decision_his_a = DecisionHistory(
            [lot_pair_0, lot_pair_1, lot_pair_2, lot_pair_3],
            [True, False, False, True],
        )
        decision_his_b = DecisionHistory(
            [lot_pair_0, lot_pair_2, lot_pair_3, lot_pair_4],
            [False, False, True, True],
        )
        self.assertTrue(
            np.allclose(
                same_decision_frac(decision_his_a, decision_his_b), 2 / 3
            )
        )

    def test_ubcfRecommender(self):
        all_lot_pairs = [lot_pair_0, lot_pair_1, lot_pair_2, lot_pair_3]
        decision_his_0 = DecisionHistory(
            all_lot_pairs, [True, False, True, False]
        )
        decision_his_1 = DecisionHistory(
            all_lot_pairs, [False, True, False, True]
        )
        decision_his_2 = DecisionHistory(
            all_lot_pairs, [True, True, False, False]
        )
        decision_his_3 = DecisionHistory(
            [lot_pair_0, lot_pair_1, lot_pair_2], [True, False, False]
        )
        records = [decision_his_0, decision_his_1, decision_his_2]
        recommender = UcbfRecommender(decision_his_3, records, 2)
        actual = recommender.rec(lot_pair_3)
        expected = False
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
