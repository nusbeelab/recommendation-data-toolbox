import numpy as np
from typing import Tuple, Union
import numpy.typing as npt


def crra_utility_fn_1param(
    params: Tuple[np.float64], x: Union[int, npt.NDArray[np.int_]]
):
    """Utility function with one parameter, U(x|r) = x^r."""
    return np.power(x, params[0], dtype=np.float64)


def crra_utility_fn_3params(
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
    utility_neg_x = -params[1] * np.power(-neg_x, params[2], dtype=np.float64)
    return utility_nonneg_x + utility_neg_x
