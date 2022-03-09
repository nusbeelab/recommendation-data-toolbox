import unittest

import numpy as np

from recommendation_data_toolbox.rec.cf.user_based import UcbfRecommender


class TestUserBased(unittest.TestCase):
    def test_ubcfRecommender(self):
        rating_matrix = np.array(
            [
                [0, 1, 0, 1, 0],
                [1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0],
                [1, 1, 1, 0, 0],
                [0, 0, 1, 0, 0],
            ]
        )
        recommender = UcbfRecommender(
            rating_matrix=rating_matrix,
            R_lot_pair_ids=np.array([1, 4, 3, 0, 2]),
            subj_lot_pair_ids=np.array([3, 2, 1]),
            decisions=np.array([0, 1, 0]),
            n_neighbors=3,
        )
        expected_rec = 0
        actual_rec = recommender.rec(0)
        self.assertEqual(actual_rec, expected_rec)


if __name__ == "__main__":
    unittest.main()
