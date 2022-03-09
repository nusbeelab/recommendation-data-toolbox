import unittest

import numpy as np

from recommendation_data_toolbox.rec.cf.neighborhood_based import (
    IbcfRecommender,
    UcbfRecommender,
)


class TestUserBased(unittest.TestCase):
    rating_matrix = None

    @classmethod
    def setUpClass(cls):
        cls.rating_matrix = np.array(
            [
                [0, 1, 0, 1, 0],
                [1, 1, 1, 1, 0],
                [0, 0, 0, 0, 0],
                [1, 1, 1, 0, 0],
                [0, 0, 1, 0, 0],
            ]
        )

    def test_ubcfRecommender(self):
        recommender = UcbfRecommender(
            rating_matrix=self.rating_matrix,
            subj_lot_pair_ids=np.array([2, 4, 0, 1]),
            subj_decisions=np.array([0, 0, 1, 1]),
            n_neighbors=3,
        )
        # 3 nearest neighbours are rows 0, 1, 3
        # their decisions for lot_pair_id == 0 are 1, 1, 0
        # thus the expected recommendation is 1
        expected_rec = 1
        actual_rec = recommender.rec(3)
        self.assertEqual(actual_rec, expected_rec)

    def test_ibcfRecommender(self):
        recommender = IbcfRecommender(
            rating_matrix=self.rating_matrix,
            subj_lot_pair_ids=np.array([2, 4, 0, 1]),
            subj_decisions=np.array([0, 0, 1, 1]),
            n_neighbors=3,
        )
        # 3 nearest neighbours are columns 0, 1, 4
        # subject's decisions for lot_pair_id == 0, 1, 4 are 1, 1, 0
        # thus the expected recommendation is 1
        expected_rec = 1
        actual_rec = recommender.rec(3)
        self.assertEqual(actual_rec, expected_rec)


if __name__ == "__main__":
    unittest.main()
