import numpy as np
from typing import TypedDict, Union
import numpy.typing as npt


class SimpleUtilityParams(TypedDict):
    r: np.float64


def get_utility_simple(
    params: SimpleUtilityParams, x: Union[int, npt.NDArray[np.int_]]
):
    return np.power(x, params["r"], dtype=np.float64)
