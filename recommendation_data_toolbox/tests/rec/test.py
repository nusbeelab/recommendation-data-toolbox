import unittest

import numpy as np

from recommendation_data_toolbox.rec import (
    MostPopularChoiceRecommender,
    NoneRecommender,
)


class TestRec(unittest.TestCase):
    def test_noneRecommender(self):
        recommender = NoneRecommender()
        actual = recommender.rec_proba(np.array([1, 5, 2]))
        self.assertTrue(np.isnan(actual).all())

    def test_mostPopularChoiceRecommender(self):
        recommender = MostPopularChoiceRecommender()
        problem_ids = np.array([63, 72, 141, 177])

        expected_probs = np.array(
            [
                0.2802690582959641,
                0.9035874439461884,
                0.3520179372197309,
                0.5807174887892377,
            ]
        )
        actual_probs = recommender.rec_proba(problem_ids)

        expected_recs = np.array([0, 1, 0, 1])
        actual_recs = recommender.rec(problem_ids)

        self.assertTrue(np.allclose(actual_probs, expected_probs))
        self.assertTrue(np.array_equal(actual_recs, expected_recs))


if __name__ == "__main__":
    unittest.main()
