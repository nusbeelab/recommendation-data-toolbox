import numpy as np
from typing import List, Literal, Optional
import numpy.typing as npt
import pandas as pd


def simplify_lottery(
    objective_consequences: npt.NDArray[np.int_], probs: npt.NDArray[np.float64]
):
    mask = probs > 0
    # consolidate objective consequences of the same values
    return objective_consequences[mask], probs[mask]


class Lottery:
    def __init__(
        self,
        objective_consequences: npt.ArrayLike,
        probs: npt.ArrayLike,
    ):
        """A container for objective consequences and nonzero probabilities of a lottery."""
        objective_consequences, probs = simplify_lottery(
            np.array(objective_consequences, dtype=np.int_),
            np.array(probs, dtype=np.float64),
        )
        self.objective_consequences = objective_consequences
        self.probs = probs
        self._sorted_ocs = None
        self._sorted_probs = None
        self._cum_probs = None

    def _sort(self):
        if self._sorted_ocs is None:
            order = self.objective_consequences.argsort()
            self._sorted_ocs = self.objective_consequences[order]
            self._sorted_probs = self.probs[order]

    def cum_prob(self, x: int):
        if self._cum_probs is None:
            self._sort()
            cum_probs = np.array(self._sorted_probs)
            for i in range(1, len(cum_probs)):
                cum_probs[i] += cum_probs[i - 1]
            self._cum_probs = cum_probs
        if x < self._sorted_ocs[0]:
            return 0
        return self._cum_probs[
            len(self._sorted_ocs) - 1 - np.argmax((self._sorted_ocs <= x)[::-1])
        ]

    def __eq__(self, o):
        return (
            isinstance(o, Lottery)
            and np.array_equal(
                self.objective_consequences, o.objective_consequences
            )
            and np.allclose(self.probs, o.probs)
        )

    def __str__(self):
        return f"objective_consequences: {self.objective_consequences}; probs: {self.probs}"


class Problem:
    def __init__(self, a: Lottery, b: Lottery):
        self.a = a
        self.b = b

    def __eq__(self, o):
        return isinstance(o, Problem) and self.a == o.a and self.b == o.b


class ProblemManager:
    def __init__(
        self, problems: List[Problem], labels: Optional[List[str]] = None
    ):
        self.problems = list(problems)
        self.labels = None if labels == None else list(labels)

    def convert_problems_to_ids(self, problems: List[Problem]):
        try:
            return [
                # manually check for equal lottery pairs instead of pre-compute
                # a hashmap so as to avoid calling hash on float attributes.
                next(
                    i for i, prob in enumerate(self.problems) if prob == problem
                )
                for problem in problems
            ]
        except StopIteration:
            raise ValueError(
                "lottery_pair is not configured in manager's store."
            )

    def convert_ids_to_problems(self, ids: List[int]):
        return [self.problems[id] for id in ids]


def decode_problem(row: pd.Series):
    lot_a = Lottery(
        np.array(row[["xa1", "xa2", "xa3"]]),
        np.array(row[["pa1", "pa2", "pa3"]]),
    )
    lot_b = Lottery(
        np.array(row[["xb1", "xb2", "xb3"]]),
        np.array(row[["pb1", "pb2", "pb3"]]),
    )
    return Problem(lot_a, lot_b)


def get_problem_manager(df: pd.DataFrame):
    return ProblemManager(list(df.apply(decode_problem, axis=1)))
