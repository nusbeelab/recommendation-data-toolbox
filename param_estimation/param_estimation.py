import os
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize
from typing import Any, Callable, List, Union
import numpy.typing as npt

from param_estimation import CWD
from param_estimation.utility_functions import (
    crra_utility_fn_1param,
    crra_utility_fn_3params,
)

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
    data: npt.NDArray[np.bool_],  # 0 if option A was chosen, 1 otherwise
):
    eu_deltas = eu_fn(params, utility_fn, a_values, a_probs) - eu_fn(
        params, utility_fn, b_values, b_probs
    )
    signs = data * -2 + 1  # 1 if option A was chosen, -1 otherwise
    return -np.log(norm.cdf(eu_deltas * signs)).sum(axis=-1)


def pad_zeros_1darray(arr: npt.NDArray[(Any,)], length: int):
    new_arr = np.zeros((length,))
    new_arr[: arr.shape[0]] = arr
    return new_arr


def stack_1darrays(arrs: List[npt.NDArray[(Any,)]]):
    """Stack 1d numpy arrays of different sizes into a 2d array, padded with zeros."""
    max_length = max([arr.shape[0] for arr in arrs])
    return np.array([pad_zeros_1darray(arr, max_length) for arr in arrs])


def decode_nparray(encoding: str, dtype: npt.DTypeLike = np.float64):
    if encoding[0] != "[" or encoding[-1] != "]":
        raise ValueError(
            'An encoded numpy array must start with "[" and end with "]"'
        )
    return np.fromstring(encoding[1:-1], dtype=dtype, sep=" ")


def get_vals_from_encodings(vals: pd.Series):
    return stack_1darrays([decode_nparray(val, dtype=int) for val in vals])


def get_probs_from_encodings(probs: pd.Series):
    return stack_1darrays([decode_nparray(prob) for prob in probs])


def estimate_params_across_df(
    df: pd.DataFrame, x0: npt.NDArray[(Any,)], utility_fn: UtilityFunc
):
    a_values = get_vals_from_encodings(df["aValues"])
    a_probs = get_probs_from_encodings(df["aProbs"])
    b_values = get_vals_from_encodings(df["bValues"])
    b_probs = get_probs_from_encodings(df["bProbs"])
    data = np.array(df["Risk"], dtype=bool)

    if utility_fn == crra_utility_fn_1param:
        mask = np.all(a_values >= 0, axis=1) & np.all(b_values >= 0, axis=1)
        a_values, a_probs, b_values, b_probs, data = (
            x[mask] for x in (a_values, a_probs, b_values, b_probs, data)
        )

    res = minimize(
        fun=neg_log_lik_fn,
        x0=x0,
        args=(utility_fn, a_values, a_probs, b_values, b_probs, data),
        method="Nelder-Mead",
    )
    return pd.Series(
        [res.success, res.fun, res.x],
        index=["isSuccess", "minNegLogLik", "optimalParams"],
    )


UTILITY_MODELS = {
    "crra1param": dict(utility_fn=crra_utility_fn_1param, params_num=1),
    "crra3params": dict(utility_fn=crra_utility_fn_3params, params_num=3),
}


def estimate_params_by_subjects(filename: str, utility_model: str):
    """Perform mle on a utility model"""
    filepath = os.path.join(CWD, "data", filename)
    df = pd.read_csv(filepath)
    return df.groupby("SubjID").apply(
        estimate_params_across_df,
        x0=np.ones((UTILITY_MODELS[utility_model]["params_num"],)),
        utility_fn=UTILITY_MODELS[utility_model]["utility_fn"],
    )
