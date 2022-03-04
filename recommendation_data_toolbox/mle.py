from typing import Callable
import numpy as np
from scipy.stats import norm
from scipy.optimize import basinhopping
import numpy.typing as npt

from recommendation_data_toolbox.models.lottery_utility import (
    get_lottery_utility_model,
)


def neg_log_lik_fn(
    params: tuple,
    lottery_utility_func: Callable[
        [tuple, npt.NDArray[np.int_], npt.NDArray[np.float64]], np.float64
    ],
    a_outcomes: npt.NDArray[np.int_],
    a_probs: npt.NDArray[np.float64],
    b_outcomes: npt.NDArray[np.int_],
    b_probs: npt.NDArray[np.float64],
    observed_data: npt.NDArray[
        np.bool_
    ],  # 0 if option A was chosen, 1 otherwise
):
    eu_deltas = np.subtract(
        lottery_utility_func(params, a_outcomes, a_probs),
        lottery_utility_func(params, b_outcomes, b_probs),
    )
    signs = observed_data * -2 + 1  # 1 if option A was chosen, -1 otherwise
    return -np.log(norm.cdf(np.multiply(eu_deltas, signs))).sum(axis=-1)


def estimate_max_lik_params(
    a_outcomes: npt.NDArray[np.int_],
    a_probs: npt.NDArray[np.float64],
    b_outcomes: npt.NDArray[np.int_],
    b_probs: npt.NDArray[np.float64],
    observed_data: npt.NDArray[np.bool_],
    lottery_utility_name: str,
    outcome_utility_name: str,
    is_with_constraints: bool = True,
):
    model = get_lottery_utility_model(
        lottery_utility_name, outcome_utility_name
    )

    return basinhopping(
        func=neg_log_lik_fn,
        x0=model.inital_params,
        minimizer_kwargs=dict(
            bounds=model.bounds if is_with_constraints else None,
            method="Nelder-Mead",
            args=(
                model.lottery_utility_func,
                a_outcomes,
                a_probs,
                b_outcomes,
                b_probs,
                observed_data,
            ),
        ),
    )
