import unittest

import numpy as np

from recommendation_data_toolbox.lottery import (
    Lottery,
    Problem,
    ProblemManager,
)
from recommendation_data_toolbox.tests.mock_data import (
    PROB_2,
    PROB_3,
    PROBLEMS,
)


class TestLottery(unittest.TestCase):
    def test_lotteryPairManager(self):
        problem_manager = ProblemManager(PROBLEMS)
        self.assertEqual(
            problem_manager.convert_ids_to_lottery_pairs([2, 3]),
            [PROB_2, PROB_3],
        )
        problem_1 = Problem(
            Lottery(np.array([4, 3, 2]), np.array([0.3, 0.2, 0.5])),
            Lottery(np.array([7, 3, 2]), np.array([0.2, 0.6, 0.2])),
        )
        problem_2 = Problem(
            Lottery(np.array([10, 3, 1]), np.array([0.2, 0.7, 0.1])),
            Lottery(np.array([11, 6, 2]), np.array([0.1, 0.5, 0.4])),
        )
        self.assertEqual(
            problem_manager.convert_lottery_pairs_to_ids(
                [problem_1, problem_2]
            ),
            [1, 2],
        )


if __name__ == "__main__":
    unittest.main()
