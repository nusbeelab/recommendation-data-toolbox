import os
import numpy as np
from scipy.stats import norm
from typing import Callable, Union
import numpy.typing as npt

from param_estimation import CWD


UtilityFunc = Callable[
    [dict, Union[np.int_, npt.NDArray[np.int_]]],
    Union[np.float64, npt.NDArray[np.float64]],
]


def get_eu(
    params: dict,
    get_utility: UtilityFunc,
    values: npt.NDArray[np.int_],
    probs: npt.NDArray[np.float64],
) -> np.float64:
    return np.sum(probs * get_utility(params, values), axis=-1)


def get_neg_log_lik(
    params: dict,
    get_utility: UtilityFunc,
    a_values: npt.NDArray[np.int_],
    a_probs: npt.NDArray[np.float64],
    b_values: npt.NDArray[np.int_],
    b_probs: npt.NDArray[np.float64],
    data: npt.NDArray[np.bool_],  # 0 if option A was chosen, 1 otherwise
):
    eu_deltas = get_eu(params, get_utility, a_values, a_probs) - get_eu(
        params, get_utility, b_values, b_probs
    )
    signs = data * -2 + 1  # 1 if option A was chosen, -1 otherwise
    return -np.log(norm.cdf(eu_deltas * signs)).sum(axis=-1)


def read_data():
    filepath = os.path.join(
        CWD, "data", "IntermediateDataForParamEstimation.csv"
    )
    # pad zeros
    pass
