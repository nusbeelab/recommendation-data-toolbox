import numpy as np
from recommendation_data_toolbox.lottery import Lottery, LotteryPair


lot_pair_0 = LotteryPair(
    Lottery(np.array([5, 3, 2]), np.array([0.1, 0.5, 0.4])),
    Lottery(np.array([6, 4, 1]), np.array([0.2, 0.7, 0.1])),
)
lot_pair_1 = LotteryPair(
    Lottery(np.array([4, 3, 2]), np.array([0.3, 0.2, 0.5])),
    Lottery(np.array([7, 3, 2]), np.array([0.2, 0.6, 0.2])),
)
lot_pair_2 = LotteryPair(
    Lottery(np.array([10, 3, 1]), np.array([0.2, 0.7, 0.1])),
    Lottery(np.array([11, 6, 2]), np.array([0.1, 0.5, 0.4])),
)
lot_pair_3 = LotteryPair(
    Lottery(np.array([11, 6, 2]), np.array([0.3, 0.2, 0.5])),
    Lottery(np.array([5, 3, 2]), np.array([0.2, 0.6, 0.2])),
)
lot_pair_4 = LotteryPair(
    Lottery(np.array([7, 3, 2]), np.array([0.2, 0.7, 0.1])),
    Lottery(np.array([6, 4, 1]), np.array([0.1, 0.5, 0.4])),
)
