import numpy as np
from typing import Tuple, Union
import numpy.typing as npt

from recommendation_data_toolbox.constants import MAX_FLOAT64


class ProbWeightModel:
    def __init__(self, prob_weight_func, bounds, initial_params):
        self.prob_weight_func = prob_weight_func
        self.bounds = bounds
        self.initial_params = initial_params


def identity_pwf(params: tuple, p: Union[np.float64, npt.NDArray[np.float64]]):
    return p


def tversky_kahneman_1992_pwf(
    params: Tuple[np.float64], p: Union[np.float64, npt.NDArray[np.float64]]
):
    """
    w(p) = p^gamma / (p^gamma + (1-p)^gamma)^(1/gamma))
    """
    gamma = params[0]
    p_to_power_gamma = np.power(p, gamma)
    one_minus_p_to_power_gamma = np.power(1 - p, gamma)
    denom = np.power(
        np.add(p_to_power_gamma, one_minus_p_to_power_gamma), 1 / gamma
    )
    return np.divide(p_to_power_gamma, denom)


PROB_WEIGHT_MODELS = {
    "expected_utility": ProbWeightModel(identity_pwf, [], []),
    "prospect_theory": ProbWeightModel(
        tversky_kahneman_1992_pwf, [(0.0, MAX_FLOAT64)], [1.0]
    ),
}


class InvalidLotteryUtilityModelName(ValueError):
    pass


def get_prob_weight_model(lottery_utility_model_name: str):
    try:
        return PROB_WEIGHT_MODELS[lottery_utility_model_name]
    except:
        raise InvalidLotteryUtilityModelName(
            f"{lottery_utility_model_name} is not a valid option for loterry utility model. Valid options: expected_utility, prospect_theory."
        )
