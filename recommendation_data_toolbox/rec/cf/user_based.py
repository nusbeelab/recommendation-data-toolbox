from typing import Any, Callable, List, Optional, TypeVar
import numpy as np

import pandas as pd
import numpy.typing as npt
from sklearn.neighbors import KNeighborsClassifier

from recommendation_data_toolbox.lottery import (
    Lottery,
    LotteryPair,
)
from recommendation_data_toolbox.rec import Recommender


class UcbfRecommender(Recommender):
    """A user-based collaborative filtering (UCBF) recommender.
    Similarity between subjects is computed by the fraction of how many lottery pairs
    are given the same response by the two subjects among their history.
    """

    def __init__(
        self,
        rating_matrix: npt.NDArray[np.int_],
        R_lot_pair_ids: npt.NDArray[np.int_],
        subj_lot_pair_ids: npt.NDArray[np.int_],
        decisions: npt.NDArray[np.int_],
        n_neighbors: Optional[int] = 5,
    ):
        """
        Parameters
        ----------
        rating_matrix : 2D-array of boolean
            A full rating matrix from pre-experiment.
        R_lottery_pair_ids : 1D-array of int
            Specifies the ids of lottery pairs that each row in rating_matrix correspond to.
        """
        self.rating_matrix = rating_matrix
        self.R_lot_pair_ids = R_lot_pair_ids
        self.subj_lot_pair_ids = subj_lot_pair_ids
        self.decisions = decisions
        self.knn_classifier = KNeighborsClassifier(
            n_neighbors=n_neighbors, weights="distance", metric="manhattan"
        )

    def rec(self, lot_pair_id: int):
        mask_X = np.isin(self.R_lot_pair_ids, self.subj_lot_pair_ids)
        X = self.rating_matrix[:, mask_X]

        mask_y = self.R_lot_pair_ids == lot_pair_id
        y = self.rating_matrix[:, mask_y][:, 0]

        self.knn_classifier.fit(X, y)
        return self.knn_classifier.predict([self.decisions])[0]
