import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from typing import Optional
import numpy.typing as npt

from recommendation_data_toolbox.rec.cf import CfRecommender


class NbcfRecommender(CfRecommender):
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
        super().__init__(rating_matrix, subj_lot_pair_ids, subj_decisions)

        self.clf = KNeighborsClassifier(
            n_neighbors=n_neighbors, weights="distance", metric="manhattan"
        )


class UbcfRecommender(NbcfRecommender):
    """A user-based collaborative filtering (UBCF) recommender."""

    def rec(self, lot_pair_id: int):
        X = self.rating_matrix[:, self.subj_lot_pair_ids]
        y = self.rating_matrix[:, lot_pair_id]

        self.clf.fit(X, y)

        return self.clf.predict([self.subj_decisions])[0]


class IbcfRecommender(NbcfRecommender):
    """An item-based collaborative filtering (IBCF) recommender."""

    def rec(self, lot_pair_id: int):
        X = self.rating_matrix[:, self.subj_lot_pair_ids].T
        y = self.subj_decisions

        self.clf.fit(X, y)

        input_X = self.rating_matrix[:, lot_pair_id]
        return self.clf.predict([input_X])[0]
