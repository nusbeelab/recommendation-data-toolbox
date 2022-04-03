import unittest

import numpy as np

from recommendation_data_toolbox.rec import (
    MostPopularChoiceRecommender,
    NoneRecommender,
)


class TestRec(unittest.TestCase):
    def test_noneRecommender(self):
        recommender = NoneRecommender()
        self.assertTrue(recommender.rec(np.array([1, 5, 2])) is None)

    def test_mostPopularChoiceRecommender(self):
        recommender = MostPopularChoiceRecommender()
        expected = np.array([0, 1, 0, 1])
        actual = recommender.rec(np.array([63, 72, 141, 177]))
        self.assertTrue(np.array_equal(actual, expected))


if __name__ == "__main__":
    unittest.main()
