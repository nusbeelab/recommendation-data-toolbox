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
        """A container for objective consequences and nonzero probabilities of a lottery.
        The objective consequences are sorted in descending order.
        """
        objective_consequences, probs = simplify_lottery(
            objective_consequences, probs
        )
        order = objective_consequences.argsort()[::-1]
        self.objective_consequences = objective_consequences[order]
        self.probs = probs[order]

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


class LotteryPair:
    def __init__(self, a: Lottery, b: Lottery):
        self.a = a
        self.b = b

    def __eq__(self, o):
        return isinstance(o, LotteryPair) and self.a == o.a and self.b == o.b


class LotteryPairManager:
    def __init__(
        self, lot_pairs: List[LotteryPair], ids: Optional[List[int]] = None
    ):
        if ids is None:
            ids = list(range(len(lot_pairs)))
        elif len(lot_pairs) != len(ids):
            raise ValueError("lottery_pairs and ids must have the same length.")
        self.lot_pair_dict = {
            id: lot_pair for lot_pair, id in zip(lot_pairs, ids)
        }

    @property
    def ids(self):
        return list(self.lot_pair_dict.keys())

    @property
    def lot_pairs(self):
        return list(self.lot_pair_dict.values())

    def convert_lottery_pairs_to_ids(self, lottery_pairs: List[LotteryPair]):
        try:
            return [
                # manually check for equal lottery pairs instead of pre-compute
                # a hashmap so as to avoid calling hash on float attributes.
                next(
                    i
                    for i, lot_pair in self.lot_pair_dict.items()
                    if lot_pair == lottery_pair
                )
                for lottery_pair in lottery_pairs
            ]
        except StopIteration:
            raise ValueError(
                "lottery_pair is not configured in manager's store."
            )

    def convert_ids_to_lottery_pairs(self, ids: List[int]):
        return [self.lot_pair_dict[id] for id in ids]
