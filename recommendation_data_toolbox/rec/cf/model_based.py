import numpy as np
import numpy.typing as npt
from recommendation_data_toolbox.rec import Recommender


class DecisionTreeRecommender(Recommender):
    def __init__(
        self,
        rating_matrix: npt.NDArray[np.int_],
    ):
        self.rating_matrix = rating_matrix
