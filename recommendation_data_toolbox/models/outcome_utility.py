import numpy as np
from typing import Callable, Dict, List, Tuple, Union
import numpy.typing as npt


OutcomeUtilityFunc = Union[
    Callable[[tuple, np.int_], np.float64],
    Callable[[tuple, npt.NDArray[np.int_]], npt.NDArray[np.float64]],
]  # params, outcome -> utility of outcome


class OutcomeUtilityModel:
    def __init__(
        self,
        outcome_utility_func: OutcomeUtilityFunc,
        bounds: List[Tuple[np.float64, np.float64]],
        initial_params: List[np.float64],
    ):
        self.outcome_utility_func = outcome_utility_func
        self.bounds = bounds
        self.initial_params = initial_params


class DomainError(ValueError):
    pass


def power_uf_on_nonneg_outcomes(
    params: Tuple[np.float64], x: Union[int, npt.NDArray[np.int_]]
):
    """Utility function with one parameter, U(x|r) = x^r."""
    if np.any(x < 0):
        raise DomainError(
            "All outcomes must be nonnegative when using a power utility function on nonnegative outcomes parameterized by one variable."
        )
    return np.power(x, params[0], dtype=np.float64)


def power_uf_on_real_outcomes(
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


OUTCOME_UTILITY_MODELS: Dict[str, OutcomeUtilityModel] = {
    "power_nonneg": OutcomeUtilityModel(
        power_uf_on_nonneg_outcomes, [(0.0, 1.0)], [0.5]
    ),
    "power": OutcomeUtilityModel(
        power_uf_on_real_outcomes,
        [(0.0, 1.0), (1.0, np.finfo(np.float64).max), (0.0, 1.0)],
        [0.5, 2.0, 0.5],
    ),
}


class InvalidOutcomeUtilityName(ValueError):
    pass


def get_outcome_utility_model(model_name: str):
    try:
        return OUTCOME_UTILITY_MODELS[model_name]
    except:
        raise InvalidOutcomeUtilityName()
