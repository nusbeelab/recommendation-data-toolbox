import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from typing import Any, Callable, Union
import numpy.typing as npt

UtilityFunc = Callable[
    [tuple, Union[np.int_, npt.NDArray[np.int_]]],
    Union[np.float64, npt.NDArray[np.float64]],
]


def eu_fn(
    params: tuple,
    utility_fn: UtilityFunc,
    values: npt.NDArray[np.int_],
    probs: npt.NDArray[np.float64],
) -> np.float64:
    return np.sum(probs * utility_fn(params, values), axis=-1)


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
    eu_deltas = eu_fn(params, utility_fn, a_values, a_probs) - eu_fn(
        params, utility_fn, b_values, b_probs
    )
    signs = observed_data * -2 + 1  # 1 if option A was chosen, -1 otherwise
    return -np.log(norm.cdf(eu_deltas * signs)).sum(axis=-1)


def mle_estimate(
    x0: npt.NDArray[(Any,)],
    utility_fn: UtilityFunc,
    a_values: npt.NDArray[np.int_],
    a_probs: npt.NDArray[np.float64],
    b_values: npt.NDArray[np.int_],
    b_probs: npt.NDArray[np.float64],
    observed_data: npt.NDArray[np.bool_],
):
    return minimize(
        fun=neg_log_lik_fn,
        x0=x0,
        args=(utility_fn, a_values, a_probs, b_values, b_probs, observed_data),
        method="Nelder-Mead",
    )
