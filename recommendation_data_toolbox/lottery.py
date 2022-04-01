import numpy as np
from typing import List, Optional
import numpy.typing as npt


def simplify_lottery(
    objective_consequences: npt.NDArray[np.int_], probs: npt.NDArray[np.float64]
):
    mask = probs > 0
    # consolidate objective consequences of the same values
    return objective_consequences[mask], probs[mask]


class Lottery:
    def __init__(
        self,
        objective_consequences: npt.NDArray[np.int_],
        probs: npt.NDArray[np.float64],
    ):
        """A container for objective consequences and nonzero probabilities of a lottery."""
        objective_consequences, probs = simplify_lottery(
            objective_consequences, probs
        )
        self.objective_consequences = objective_consequences
        self.probs = probs

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

    def convert_lottery_pairs_to_ids(self, problems: List[Problem]):
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

    def convert_ids_to_lottery_pairs(self, ids: List[int]):
        return [self.problems[id] for id in ids]
