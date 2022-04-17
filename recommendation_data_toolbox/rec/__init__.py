from abc import ABCMeta, abstractmethod
import importlib.resources
import numpy as np
import pandas as pd

import numpy.typing as npt


class Recommender(metaclass=ABCMeta):
    @abstractmethod
    def rec_proba(
        self, problem_ids: npt.NDArray[np.int_]
    ) -> npt.NDArray[np.float64]:
        pass

    def rec(self, problem_ids: npt.NDArray[np.int_]):
        probs = self.rec_proba(problem_ids)
        recs = np.empty(probs.shape)
        recs[:] = np.nan
        mask = ~np.isnan(probs)
        recs[mask] = probs[mask] >= 0.5
        return recs


class NoneRecommender(Recommender):
    def rec_proba(self, problem_ids: npt.NDArray[np.int_]):
        return np.array([None for _ in problem_ids], dtype=np.float64)


class RandomRecommender(Recommender):
    def rec_proba(self, problem_ids: npt.NDArray[np.int_]):
        return np.random.rand(len(problem_ids))


class MostPopularChoiceRecommender(Recommender):
    def __init__(self):
        with importlib.resources.path(
            "recommendation_data_toolbox.resources",
            "Preexperiment_OptionBFracs.csv",
        ) as file:
            df = pd.read_csv(file)
        self._optionB_fracs = {
            int(row["problem_id"]): row["optionB_frac"]
            for _, row in df.iterrows()
        }

    def rec_proba(self, problem_ids: npt.NDArray[np.int_]):
        return np.array(
            [self._optionB_fracs.get(id, np.nan) for id in problem_ids],
            dtype=np.float64,
        )
