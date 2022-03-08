from typing import Callable, List, Optional, TypeVar

from recommendation_data_toolbox.lottery import DecisionHistory, LotteryPair
from recommendation_data_toolbox.rec import Recommender


def same_decision_frac(a: DecisionHistory, b: DecisionHistory):
    common_lottery_pairs = [
        lot_pair for lot_pair in a.lottery_pairs if lot_pair in b.lottery_pairs
    ]  # O(n^2)
    same_decision_num = sum(a[lot] == b[lot] for lot in common_lottery_pairs)
    return same_decision_num / len(common_lottery_pairs)


T = TypeVar("T")


def get_nearest_neighbors(
    records: List[T],
    query: T,
    k: int,
    metric: Callable[[T, T], float],
):
    if k > len(records):
        raise ValueError("k must not be greater than the number of records.")
    dists = [metric(record, query) for record in records]
    knn_indices = sorted(range(len(dists)), key=lambda x: dists[x])[-k:]
    return [records[i] for i in knn_indices]


class UcbfRecommender(Recommender):
    """A user-based collaborative filtering (UCBF) recommender.
    Similarity between subjects is computed by the fraction of how many lottery pairs
    are given the same response by the two subjects among their history.
    """

    def __init__(
        self,
        history: DecisionHistory,
        records: List[DecisionHistory],
        k: Optional[int],
    ):
        if k == None:
            k = len(records) % 3
        if k > len(records):
            raise ValueError(
                "k must not be greater than the number of records."
            )
        self.records = records
        self.history = history
        self.k = k
        self.knns = None

    def rec(self, lottery_pair: LotteryPair):
        if self.knns == None:
            self.knns = get_nearest_neighbors(
                records=self.records,
                query=self.history,
                k=self.k,
                metric=same_decision_frac,
            )
        decisions_in_records = [his[lottery_pair] for his in self.knns]
        return sum(decisions_in_records) / len(decisions_in_records) >= 0.5
