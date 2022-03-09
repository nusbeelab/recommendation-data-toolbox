from typing import Optional
import numpy as np

import numpy.typing as npt
from sklearn.neighbors import KNeighborsClassifier

from recommendation_data_toolbox.rec import Recommender


class UcbfRecommender(Recommender):
    """A user-based collaborative filtering (UBCF) recommender."""

    def __init__(
        self,
        rating_matrix: npt.NDArray[np.int_],
        subj_lot_pair_ids: npt.NDArray[np.int_],
        subj_decisions: npt.NDArray[np.int_],
        n_neighbors: Optional[int] = 5,
    ):
        """
        Parameters
        ----------
        rating_matrix : 2D-array of boolean
            A full rating matrix from pre-experiment. Each row represents the decisions
            of a subject for lot_pair_ids 0..(n-1).
        """
        self.rating_matrix = rating_matrix

        order = subj_lot_pair_ids.argsort()
        self.subj_lot_pair_ids = subj_lot_pair_ids[order]
        self.subj_decisions = subj_decisions[order]

        self.knn_classifier = KNeighborsClassifier(
            n_neighbors=n_neighbors, weights="distance", metric="manhattan"
        )

    def rec(self, lot_pair_id: int):
        X = self.rating_matrix[:, self.subj_lot_pair_ids]
        y = self.rating_matrix[:, lot_pair_id]

        self.knn_classifier.fit(X, y)

        return self.knn_classifier.predict([self.subj_decisions])[0]


class IbcfRecommender(Recommender):
    def __init__(
        self,
        rating_matrix: npt.NDArray[np.int_],
        subj_lot_pair_ids: npt.NDArray[np.int_],
        subj_decisions: npt.NDArray[np.int_],
        n_neighbors: Optional[int] = 5,
    ):
        """
        Parameters
        ----------
        rating_matrix : 2D-array of boolean
            A full rating matrix from pre-experiment. Each row represents the decisions
            of a subject for lot_pair_ids 0..(n-1).
        """
        self.rating_matrix = rating_matrix

        order = subj_lot_pair_ids.argsort()
        self.subj_lot_pair_ids = subj_lot_pair_ids[order]
        self.subj_decisions = subj_decisions[order]

        self.knn_classifier = KNeighborsClassifier(
            n_neighbors=n_neighbors, weights="distance", metric="manhattan"
        )

    def rec(self, lot_pair_id: int):
        X = self.rating_matrix[:, self.subj_lot_pair_ids].T
        y = self.subj_decisions

        self.knn_classifier.fit(X, y)

        input_X = self.rating_matrix[:, lot_pair_id]
        return self.knn_classifier.predict([input_X])
