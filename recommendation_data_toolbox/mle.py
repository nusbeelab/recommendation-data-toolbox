from typing import Callable, Optional
import numpy as np
from scipy.stats import norm
from scipy.optimize import basinhopping
import numpy.typing as npt

from recommendation_data_toolbox.lottery_utility import get_lottery_utility


def neg_log_lik_fn(
    params: tuple,
    lottery_utility_func: Callable[
        [tuple, npt.NDArray[np.int_], npt.NDArray[np.float64]], np.float64
    ],
    a_ocs: npt.NDArray[np.int_],
    a_probs: npt.NDArray[np.float64],
    b_ocs: npt.NDArray[np.int_],
    b_probs: npt.NDArray[np.float64],
    observed_data: npt.NDArray[
        np.int_
    ],  # 0 if option A was chosen, 1 otherwise
):
    eu_deltas = np.subtract(
        lottery_utility_func(params, a_ocs, a_probs),
        lottery_utility_func(params, b_ocs, b_probs),
    )
    signs = observed_data * -2 + 1  # 1 if option A was chosen, -1 otherwise
    return -np.log(norm.cdf(np.multiply(eu_deltas, signs))).sum(axis=-1)


def estimate_max_lik_params(
    a_ocs: npt.NDArray[np.int_],
    a_probs: npt.NDArray[np.float64],
    b_ocs: npt.NDArray[np.int_],
    b_probs: npt.NDArray[np.float64],
    observed_data: npt.NDArray[np.int_],
    lottery_utility: str,
    outcome_utility: str,
    initial_params: Optional[tuple] = None,
    is_with_constraints: bool = True,
):
    func, bounds, default_initial_params = get_lottery_utility(
        lottery_utility, outcome_utility
    )

    return basinhopping(
        func=neg_log_lik_fn,
        x0=initial_params or default_initial_params,
        minimizer_kwargs=dict(
            bounds=bounds if is_with_constraints else None,
            method="Nelder-Mead",
            args=(
                func,
                a_ocs,
                a_probs,
                b_ocs,
                b_probs,
                observed_data,
            ),
        ),
    )
