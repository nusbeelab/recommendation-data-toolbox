import numpy as np
from typing import Callable, List, Tuple
import numpy.typing as npt

from recommendation_data_toolbox.constants import MAX_FLOAT64

ProbWeightFunc = Callable[
    [tuple, npt.NDArray[np.float64]], npt.NDArray[np.float64]
]  # params, probs -> probs


class ProbWeightModel:
    def __init__(
        self,
        prob_weight_func: ProbWeightFunc,
        bounds: List[Tuple[np.float64, np.float64]],
        initial_params: List[np.float64],
    ):
        self.prob_weight_func = prob_weight_func
        self.bounds = bounds
        self.initial_params = initial_params


def identity_pwf(params: tuple, probs: npt.NDArray[np.float64]):
    """
    Parameters
    ----------
    params : tuple
        should be empty.
    probs : numpy array of float64
        Probabilities of a lottery, adding up to 1.

    Returns
    -------
    Probabilities of the lottery, unchanged.
    """
    return probs


def tk1992_pt_pwf(params: Tuple[np.float64], probs: npt.NDArray[np.float64]):
    """
    Tversky and Kahneman, 1992, for prospect theory model.
    w(p) = p^gamma / (p^gamma + (1-p)^gamma)^(1/gamma))

    Parameters
    ----------
    params : tuple
        holds (gamma,).
    probs : numpy array of float64
        Probabilities of a lottery, adding up to 1.

    Returns
    -------
    Weighted probabilities of the lottery according to prospect theory.
    """
    gamma = params[0]
    p_to_power_gamma = np.power(probs, gamma)
    one_minus_p_to_power_gamma = np.power(np.subtract(1, probs), gamma)
    denom = np.power(
        np.add(p_to_power_gamma, one_minus_p_to_power_gamma), 1 / gamma
    )
    return np.divide(p_to_power_gamma, denom)


def roll_fill_last_dim(arr: np.ndarray, shift, fill_value=0.0):
    result = np.empty_like(arr)
    if shift > 0:
        result[..., :shift] = fill_value
        result[..., shift:] = arr[..., :-shift]
    elif shift < 0:
        result[..., shift:] = fill_value
        result[..., :shift] = arr[..., -shift:]
    else:
        result[...] = arr
    return result


def tk1992_cpt_pwf(params: Tuple[np.float64], probs: npt.NDArray[np.float64]):
    """
    pi_i = w(p_i + ... + p_n) - w(p_{i+1} + ... + p_n)
    Parameters
    ----------
    params : tuple
        holds (gamma,).
    probs : numpy array of float64
        Probabilities of a lottery, adding up to 1.

    Returns
    -------
    Weighted probabilities of the lottery according to cummulative prospect theory.
    """
    sums = sum(roll_fill_last_dim(probs, -i) for i in range(probs.shape[-1]))
    weights = tk1992_pt_pwf(params, sums)
    marginal_weights = np.subtract(weights, roll_fill_last_dim(weights, -1))
    return marginal_weights


PROB_WEIGHT_MODELS = {
    "expected_utility": ProbWeightModel(identity_pwf, [], []),
    "prospect_theory": ProbWeightModel(
        tk1992_pt_pwf, [(0.0, MAX_FLOAT64)], [1.0]
    ),
    "cumulative_prospect_theory": ProbWeightModel(
        tk1992_cpt_pwf, [(0.0, MAX_FLOAT64)], [1.0]
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
