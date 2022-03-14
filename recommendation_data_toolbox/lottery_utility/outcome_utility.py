import numpy as np
from typing import Callable, List, Optional, Tuple, Union
import numpy.typing as npt
from recommendation_data_toolbox.constants import MAX_FLOAT64

from recommendation_data_toolbox.lottery_utility.exceptions import DomainError


OutcomeUtilityFunc = Union[
    Callable[[tuple, np.int_], np.float64],
    Callable[[tuple, npt.NDArray[np.int_]], npt.NDArray[np.float64]],
]  # params, outcome -> utility of outcome


def power_uf_on_nonneg_ocs(
    params: Tuple[np.float64], x: Union[int, npt.NDArray[np.int_]]
):
    """Utility function with one parameter, U(x|r) = x^r."""
    if np.any(x < 0):
        raise DomainError(
            """All object consequences must be nonnegative when using a power utility 
            function on nonnegative objective consequences parameterized by one variable.
            """
        )
    return np.power(x, params[0], dtype=np.float64)


def power_uf_on_real_ocs(
    params: Tuple[np.float64, np.float64, np.float64],
    x: Union[int, npt.NDArray[np.int_]],
):
    """Utility function with three parameters,
    U(x|r,lamba,theta) = x^r if x >= 0,
                        -lambda * (-x)^theta otherwise
    Args:
        params: a tuple of (r, lambda, theta)
    """
    nonneg_x = np.where(x >= 0, x, 0)
    utility_nonneg_x = np.power(nonneg_x, params[0], dtype=np.float64)
    neg_x = np.where(x < 0, x, 0)
    utility_neg_x = np.multiply(
        -params[1], np.power(-neg_x, params[2], dtype=np.float64)
    )
    return np.add(utility_nonneg_x, utility_neg_x)


class OutcomeUtilityWrapper:
    def __init__(self, func, bounds, initial_params):
        self.func = func
        self.bounds = bounds
        self.initial_params = initial_params


OUTCOME_UTILITY_WRAPPERS = {
    "power_nonneg": OutcomeUtilityWrapper(
        power_uf_on_nonneg_ocs, [(0.0, 1.0)], [0.5]
    ),
    "power": OutcomeUtilityWrapper(
        power_uf_on_real_ocs,
        [(0.0, 1.0), (1.0, MAX_FLOAT64), (0.0, 1.0)],
        [0.5, 2.0, 0.5],
    ),
}


# class OutcomeUtility:
#     def __init__(self, func: str, params: Optional[List[np.float64]] = None):
#         """
#         Parameters
#         ---------
#         func : str
#             Either "power" or "power_nonneg"
#         params : list
#             Values for the parameters.
#         """
#         try:
#             self.func = OUTCOME_UTILITY_WRAPPERS[func].func
#             self.params = params
#         except:
#             raise InvalidOutcomeUtilityName()

#     def compute(self, x: Union[int, npt.NDArray[np.int_]]):
#         return self.func(self.params, x)


def get_outcome_utility_func(outcome_utility: str):
    return OUTCOME_UTILITY_WRAPPERS[outcome_utility].func


def get_outcome_utility_bounds(outcome_utility: str):
    return OUTCOME_UTILITY_WRAPPERS[outcome_utility].bounds


def get_outcome_utility_initial_params(outcome_utility: str):
    return OUTCOME_UTILITY_WRAPPERS[outcome_utility].initial_params
