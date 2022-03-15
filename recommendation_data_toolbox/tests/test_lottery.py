import unittest

import numpy as np

from recommendation_data_toolbox.lottery import (
    Lottery,
    LotteryPair,
    LotteryPairManager,
)
from recommendation_data_toolbox.tests.mock_data import (
    LOT_PAIR_2,
    LOT_PAIR_3,
    LOT_PAIRS,
)


class TestLottery(unittest.TestCase):
    def test_lottery(self):
        lottery = Lottery(np.array([3, 5, 2]), np.array([0.1, 0.5, 0.4]))
        self.assertTrue(
            np.array_equal(lottery.objective_consequences, np.array([5, 3, 2]))
        )
        self.assertTrue(
            np.array_equal(lottery.probs, np.array([0.5, 0.1, 0.4]))
        )

    def test_lotteryPairManager_properties(self):
        lot_pair_manager = LotteryPairManager(LOT_PAIRS, [5, 3, 4, 1, 2])
        self.assertEqual(lot_pair_manager.lot_pairs, LOT_PAIRS)
        self.assertEqual(lot_pair_manager.ids, [5, 3, 4, 1, 2])

    def test_lotteryPairManager_noIds(self):
        lot_pair_manager = LotteryPairManager(LOT_PAIRS)
        self.assertEqual(
            lot_pair_manager.convert_ids_to_lottery_pairs([2, 3]),
            [LOT_PAIR_2, LOT_PAIR_3],
        )
        lot_pair_1 = LotteryPair(
            Lottery(np.array([4, 3, 2]), np.array([0.3, 0.2, 0.5])),
            Lottery(np.array([7, 3, 2]), np.array([0.2, 0.6, 0.2])),
        )
        lot_pair_2 = LotteryPair(
            Lottery(np.array([10, 3, 1]), np.array([0.2, 0.7, 0.1])),
            Lottery(np.array([11, 6, 2]), np.array([0.1, 0.5, 0.4])),
        )
        self.assertEqual(
            lot_pair_manager.convert_lottery_pairs_to_ids(
                [lot_pair_1, lot_pair_2]
            ),
            [1, 2],
        )

    def test_lotteryPairManager(self):
        lot_pair_manager = LotteryPairManager(LOT_PAIRS, [5, 3, 4, 1, 2])
        self.assertEqual(
            lot_pair_manager.convert_ids_to_lottery_pairs([4, 1]),
            [LOT_PAIR_2, LOT_PAIR_3],
        )
        lot_pair_1 = LotteryPair(
            Lottery(np.array([4, 3, 2]), np.array([0.3, 0.2, 0.5])),
            Lottery(np.array([7, 3, 2]), np.array([0.2, 0.6, 0.2])),
        )
        lot_pair_2 = LotteryPair(
            Lottery(np.array([10, 3, 1]), np.array([0.2, 0.7, 0.1])),
            Lottery(np.array([11, 6, 2]), np.array([0.1, 0.5, 0.4])),
        )
        self.assertEqual(
            lot_pair_manager.convert_lottery_pairs_to_ids(
                [lot_pair_1, lot_pair_2]
            ),
            [3, 4],
        )


if __name__ == "__main__":
    unittest.main()
