import numpy as np
from recommendation_data_toolbox.lottery import Lottery, Problem


PROB_0 = Problem(
    Lottery(np.array([5, 3, 2]), np.array([0.1, 0.5, 0.4])),
    Lottery(np.array([6, 4, 1]), np.array([0.2, 0.7, 0.1])),
)
PROB_1 = Problem(
    Lottery(np.array([4, 3, 2]), np.array([0.3, 0.2, 0.5])),
    Lottery(np.array([7, 3, 2]), np.array([0.2, 0.6, 0.2])),
)
PROB_2 = Problem(
    Lottery(np.array([10, 3, 1]), np.array([0.2, 0.7, 0.1])),
    Lottery(np.array([11, 6, 2]), np.array([0.1, 0.5, 0.4])),
)
PROB_3 = Problem(
    Lottery(np.array([11, 6, 2]), np.array([0.3, 0.2, 0.5])),
    Lottery(np.array([5, 3, 2]), np.array([0.2, 0.6, 0.2])),
)
PROB_4 = Problem(
    Lottery(np.array([7, 3, 2]), np.array([0.2, 0.7, 0.1])),
    Lottery(np.array([6, 4, 1]), np.array([0.1, 0.5, 0.4])),
)

PROBLEMS = [PROB_0, PROB_1, PROB_2, PROB_3, PROB_4]
