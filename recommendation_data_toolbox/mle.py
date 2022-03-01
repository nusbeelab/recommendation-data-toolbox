import numpy as np
from scipy.stats import norm
from scipy.optimize import basinhopping
import random
from typing import List, Tuple
import numpy.typing as npt

from recommendation_data_toolbox.utility_functions import (
    UtilityFunc,
    UtilityFuncWrapper,
)


def eu_fn(
    params: tuple,
    utility_fn: UtilityFunc,
    values: npt.NDArray[np.int_],
    probs: npt.NDArray[np.float64],
) -> np.float64:
    return np.sum(np.multiply(probs, utility_fn(params, values)), axis=-1)


def neg_log_lik_fn(
    params: tuple,
    utility_fn: UtilityFunc,
    a_values: npt.NDArray[np.int_],
    a_probs: npt.NDArray[np.float64],
    b_values: npt.NDArray[np.int_],
    b_probs: npt.NDArray[np.float64],
    observed_data: npt.NDArray[
        np.bool_
    ],  # 0 if option A was chosen, 1 otherwise
):
    eu_deltas = np.subtract(
        eu_fn(params, utility_fn, a_values, a_probs),
        eu_fn(params, utility_fn, b_values, b_probs),
    )
    signs = observed_data * -2 + 1  # 1 if option A was chosen, -1 otherwise
    return -np.log(norm.cdf(np.multiply(eu_deltas, signs))).sum(axis=-1)


def get_initial_params(bounds: List[Tuple[np.float64, np.float64]]):
    return np.array([random.uniform(*bound) for bound in bounds])


def estimate_max_lik_params(
    utility_func_wrapper: UtilityFuncWrapper,
    a_values: npt.NDArray[np.int_],
    a_probs: npt.NDArray[np.float64],
    b_values: npt.NDArray[np.int_],
    b_probs: npt.NDArray[np.float64],
    observed_data: npt.NDArray[np.bool_],
):
    return basinhopping(
        func=neg_log_lik_fn,
        x0=utility_func_wrapper.get_random_params(),
        minimizer_kwargs=dict(
            bounds=utility_func_wrapper.bounds,
            method="Nelder-Mead",
            args=(
                utility_func_wrapper.utility_fn,
                a_values,
                a_probs,
                b_values,
                b_probs,
                observed_data,
            ),
        ),
    )
