import numpy as np
from typing import Tuple, Union
import numpy.typing as npt


def utility_fn_1param(
    params: Tuple[np.float64], x: Union[int, npt.NDArray[np.int_]]
):
    """Utility function with one parameter, U(x|r) = x^r."""
    return np.power(x, params[0], dtype=np.float64)
