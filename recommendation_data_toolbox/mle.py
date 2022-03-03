from typing import Callable, List
import numpy as np
from scipy.stats import norm
from scipy.optimize import basinhopping
import numpy.typing as npt

from recommendation_data_toolbox.lottery import Lottery
from recommendation_data_toolbox.models.lottery_utility import (
    get_lottery_utility_model,
)
from recommendation_data_toolbox.utils import stack_1darrays


def stack_outcomes_and_probs(lotteries: List[Lottery]):
    return stack_1darrays(
        [lottery.outcomes for lottery in lotteries]
    ), stack_1darrays([lottery.probs for lottery in lotteries])


def neg_log_lik_fn(
    params: tuple,
    lottery_utility_func: Callable[
        [tuple, npt.NDArray[np.int_], npt.NDArray[np.float64]], np.float64
    ],
    a_lotteries: List[Lottery],
    b_lotteries: List[Lottery],
    observed_data: npt.NDArray[
        np.bool_
    ],  # 0 if option A was chosen, 1 otherwise
):
    eu_deltas = np.subtract(
        lottery_utility_func(params, *stack_outcomes_and_probs(a_lotteries)),
        lottery_utility_func(params, *stack_outcomes_and_probs(b_lotteries)),
    )
    signs = observed_data * -2 + 1  # 1 if option A was chosen, -1 otherwise
    return -np.log(norm.cdf(np.multiply(eu_deltas, signs))).sum(axis=-1)


def estimate_max_lik_params(
    a_lottery: Lottery,
    b_lottery: Lottery,
    observed_data: npt.NDArray[np.bool_],
    lottery_utility_name: str,
    outcome_utility_name: str = "power_nonneg",
):
    model = get_lottery_utility_model(
        lottery_utility_name, outcome_utility_name
    )

    return basinhopping(
        func=neg_log_lik_fn,
        x0=model.inital_params,
        minimizer_kwargs=dict(
            bounds=model.bounds,
            method="Nelder-Mead",
            args=(
                model.lottery_utility_func,
                a_lottery,
                b_lottery,
                observed_data,
            ),
        ),
    )
