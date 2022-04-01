from abc import ABCMeta, abstractmethod
import numpy as np
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
        return None


class RandomRecommender(Recommender):
    def rec(self, problem_ids: npt.NDArray[np.int_]):
        return np.random.randint(2, size=len(problem_ids))
