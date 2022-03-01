import random
import numpy as np
from typing import Callable, Dict, List, Tuple, Union
import numpy.typing as npt


UtilityFunc = Callable[
    [tuple, Union[np.int_, npt.NDArray[np.int_]]],
    Union[np.float64, npt.NDArray[np.float64]],
]


class UtilityFuncWrapper:
    def __init__(
        self,
        utility_fn: UtilityFunc,
        bounds: List[Tuple[np.float64, np.float64]],
    ):
        self.utility_fn = utility_fn
        self.bounds = bounds

    def get_random_params(self):
        return np.array([random.uniform(*bound) for bound in self.bounds])


def crra1param_utility_fn(
    params: Tuple[np.float64], x: Union[int, npt.NDArray[np.int_]]
):
    """Utility function with one parameter, U(x|r) = x^r."""
    return np.power(x, params[0], dtype=np.float64)


def crra3params_utility_fn(
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


UTILITY_FUNC_WRAPPERS: Dict[str, UtilityFuncWrapper] = {
    "crra1param": UtilityFuncWrapper(crra1param_utility_fn, [(0.0, 1.0)]),
    "crra3params": UtilityFuncWrapper(
        crra3params_utility_fn,
        [(0.0, 1.0), (1.0, np.finfo(np.float64).max), (0.0, 1.0)],
    ),
}


def get_utility_func_wrapper(name: str):
    return UTILITY_FUNC_WRAPPERS.get(name)
