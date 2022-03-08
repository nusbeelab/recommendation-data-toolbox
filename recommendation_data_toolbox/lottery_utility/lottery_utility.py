import numpy as np
import numpy.typing as npt

from .outcome_utility import (
    get_outcome_utility_bounds,
    get_outcome_utility_func,
    get_outcome_utility_initial_params,
)
from .prob_weight import (
    get_prob_weight_bounds,
    get_prob_weight_func,
    get_prob_weight_initial_params,
)


def get_lottery_utility_func(
    lottery_utility: str, outcome_utility: str = "power_nonneg"
):
    outcome_utility_func = get_outcome_utility_func(outcome_utility)
    prob_weight_func = get_prob_weight_func(lottery_utility)

    outcome_utility_param_num = len(
        get_outcome_utility_initial_params(outcome_utility)
    )

    def lottery_utility_func(
        params: tuple,
        outcomes: npt.NDArray[np.int_],
        probs: npt.NDArray[np.float64],
    ):
        utility_of_outcomes = outcome_utility_func(
            params[:outcome_utility_param_num], outcomes
        )
        prob_weights = prob_weight_func(
            params[outcome_utility_param_num:], probs
        )
        return np.sum(np.multiply(prob_weights, utility_of_outcomes), axis=-1)

    return lottery_utility_func


def get_lottery_utility_bounds(
    lottery_utility: str, outcome_utility: str = "power_nonneg"
):
    return get_outcome_utility_bounds(outcome_utility) + get_prob_weight_bounds(
        lottery_utility
    )


def get_lottery_utility_initial_params(
    lottery_utility: str, outcome_utility: str = "power_nonneg"
):
    return get_outcome_utility_initial_params(
        outcome_utility
    ) + get_prob_weight_initial_params(lottery_utility)


def get_lottery_utility(
    lottery_utility: str, outcome_utility: str = "power_nonneg"
):
    return tuple(
        f(lottery_utility, outcome_utility)
        for f in [
            get_lottery_utility_func,
            get_lottery_utility_bounds,
            get_lottery_utility_initial_params,
        ]
    )
