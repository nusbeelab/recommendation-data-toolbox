from abc import ABCMeta, abstractmethod
import numpy as np
import pandas as pd
import importlib.resources

from typing import Optional
import numpy.typing as npt


class Recommender(metaclass=ABCMeta):
    @abstractmethod
    def rec(
        self, problem_ids: npt.NDArray[np.int_]
    ) -> Optional[npt.NDArray[np.bool_]]:
        pass


class NoneRecommender(Recommender):
    def rec(self, problem_ids: npt.NDArray[np.int_]):
        return np.array([None for _ in problem_ids])


class RandomRecommender(Recommender):
    def rec(self, problem_ids: npt.NDArray[np.int_]):
        return np.random.randint(2, size=len(problem_ids))


class MostPopularChoiceRecommender(Recommender):
    def __init__(self):
        with importlib.resources.path(
            "recommendation_data_toolbox.resources", "majority.csv"
        ) as file:
            df = pd.read_csv(file)
        self.most_popular_choice_dict = {
            (row["stage"] - 1) * 60
            + row["problem"]
            - 1: row["most popular choice"]
            for _, row in df.iterrows()
        }

    def rec(self, problem_ids: npt.NDArray[np.int_]):
        return np.array(
            [self.most_popular_choice_dict[id] for id in problem_ids]
        )
