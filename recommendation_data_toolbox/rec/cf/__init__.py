import numpy as np
import numpy.typing as npt

from recommendation_data_toolbox.rec import Recommender


class CfRecommender(Recommender):
    """Base class for collaborative filtering recommenders."""

    def __init__(
        self,
        rating_matrix: npt.NDArray[np.int_],
        subj_problem_ids: npt.NDArray[np.int_],
        subj_decisions: npt.NDArray[np.int_],
    ):
        self.rating_matrix = rating_matrix

        order = subj_problem_ids.argsort()
        self.subj_problem_ids = subj_problem_ids[order]
        self.subj_decisions = subj_decisions[order]
