from typing import List, Optional

import numpy as np
from recommendation_data_toolbox.lottery import (
    DecisionHistory,
    Lottery,
    Problem,
)
from recommendation_data_toolbox.lottery_utility import get_lottery_utility_func
from recommendation_data_toolbox.mle import estimate_max_lik_params
from recommendation_data_toolbox.rec import Recommender
from recommendation_data_toolbox.utils import stack_1darrays


def stack_ocs_and_probs(lotteries: List[Lottery]):
    ocs = stack_1darrays([lot.objective_consequences for lot in lotteries])
    probs = stack_1darrays([lot.probs for lot in lotteries])
    return ocs, probs


def stack_problems(problems: List[Problem]):
    a_problems = [prob.a for prob in problems]
    b_problems = [prob.b for prob in problems]
    return *stack_ocs_and_probs(a_problems), *stack_ocs_and_probs(b_problems)


class LotteryUtilityRecommender(Recommender):
    def __init__(
        self,
        lottery_utility: str,
        outcome_utility: str = "nonneg",
        params: Optional[tuple] = None,
    ):
        self.lottery_utility = lottery_utility
        self.outcome_utility = outcome_utility
        self.lottery_utility_func = get_lottery_utility_func(
            self.lottery_utility, self.outcome_utility
        )
        self.params = params

    def fit(self, his: DecisionHistory):
        a_ocs, a_probs, b_ocs, b_probs = stack_problems(his.lottery_pairs)
        res = estimate_max_lik_params(
            a_ocs=a_ocs,
            a_probs=a_probs,
            b_ocs=b_ocs,
            b_probs=b_probs,
            observed_data=np.array(his.decisions),
            lottery_utility=self.lottery_utility,
            outcome_utility=self.outcome_utility,
            initial_params=self.params,
        )
        if res.success:
            self.params = res.x

    def get_utility(self, lot: Lottery):
        if self.params == None:
            raise Exception
        return self.lottery_utility_func(
            self.params, lot.objective_consequences, lot.probs
        )

    def rec(self, lottery_pair: Problem):
        return self.get_utility(lottery_pair.a) < self.get_utility(
            lottery_pair.b
        )


class EuRecommender(LotteryUtilityRecommender):
    def __init__(
        self, outcome_utility: str = "nonneg", params: Optional[tuple] = None
    ):
        super().__init__("expected_utility", outcome_utility, params)


class PtRecommender(LotteryUtilityRecommender):
    def __init__(
        self, outcome_utility: str = "nonneg", params: Optional[tuple] = None
    ):
        super().__init__("prospect_theory", outcome_utility, params)


class CptRecommender(LotteryUtilityRecommender):
    def __init__(
        self, outcome_utility: str = "nonneg", params: Optional[tuple] = None
    ):
        super().__init__("cumulative_prospect_theory", outcome_utility, params)
