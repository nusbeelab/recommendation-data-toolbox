import numpy as np
from typing import List
import numpy.typing as npt


def simplify_lottery(
    outcomes: npt.NDArray[np.int_], probs: npt.NDArray[np.float64]
):
    mask = probs > 0
    # consolidate outcomes of the same values
    return outcomes[mask], probs[mask]


class Lottery:
    def __init__(
        self, outcomes: npt.NDArray[np.int_], probs: npt.NDArray[np.float64]
    ):
        """A container for outcomes and nonzero probabilities of a lottery.
        The outcomes are sorted in descending order.
        """
        outcomes, probs = simplify_lottery(outcomes, probs)
        order = outcomes.argsort()[::-1]
        self.outcomes = outcomes[order]
        self.probs = probs[order]

    def __eq__(self, o):
        return (
            isinstance(o, Lottery)
            and np.array_equal(self.outcomes, o.outcomes)
            and np.allclose(self.probs, o.probs)
        )

    def __str__(self):
        return f"Outcomes: {self.outcomes}; Probs: {self.probs}"


class LotteryPair:
    def __init__(self, a: Lottery, b: Lottery):
        self.a = a
        self.b = b

    def __eq__(self, o):
        return isinstance(o, LotteryPair) and self.a == o.a and self.b == o.b


class LotteryPairManager:
    def __init__(self, lottery_pairs: List[LotteryPair]):
        self.lottery_pairs = list(lottery_pairs)

    def convert_lottery_pairs_to_ids(self, lottery_pairs: List[LotteryPair]):
        try:
            return [
                # manually check for equal lottery pairs instead of pre-compute
                # a hashmap so as to avoid calling hash on float attributes.
                next(
                    i
                    for i, lot_pair in enumerate(self.lottery_pairs)
                    if lot_pair == lottery_pair
                )
                for lottery_pair in lottery_pairs
            ]
        except StopIteration:
            raise ValueError(
                "lottery_pair is not configured in manager's store."
            )

    def convert_ids_to_lottery_pairs(self, ids: List[int]):
        return [self.lottery_pairs[id] for id in ids]
