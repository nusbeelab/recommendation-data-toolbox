import numpy as np
from typing import Callable, List, Optional, Tuple
import numpy.typing as npt

from recommendation_data_toolbox.constants import MAX_FLOAT64
from recommendation_data_toolbox.lottery_utility.utils import roll_fill_last_dim

ProbWeightFunc = Callable[
    [tuple, npt.NDArray[np.float64]], npt.NDArray[np.float64]
]  # params, probs -> probs


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


class ProbWeightWrapper:
    def __init__(self, func, bounds, initial_params):
        self.func = func
        self.bounds = bounds
        self.initial_params = initial_params


PROB_WEIGHT_WRAPPERS = {
    "expected_utility": ProbWeightWrapper(identity_pwf, [], []),
    "prospect_theory": ProbWeightWrapper(
        tk1992_pt_pwf, [(0.0, MAX_FLOAT64)], [1.0]
    ),
    "cumulative_prospect_theory": ProbWeightWrapper(
        tk1992_cpt_pwf, [(0.0, MAX_FLOAT64)], [1.0]
    ),
}


# class ProbWeight:
#     def __init__(self, func: str, params: Optional[List[np.float64]] = None):
#         """
#         Parameters
#         ----------
#         func : str
#             Must be "expected_utility", "prospect_theory", or "cumulative_prospect_theory".
#         params : list
#             Values for the parameters.
#         """
#         self.func = PROB_WEIGHTS[func].func
#         self.params = params

#     def compute(self, probs: npt.NDArray[np.float64]):
#         return self.func(self.params, probs)


def get_prob_weight_func(prob_weight: str):
    return PROB_WEIGHT_WRAPPERS[prob_weight].func


def get_prob_weight_bounds(prob_weight: str):
    return PROB_WEIGHT_WRAPPERS[prob_weight].bounds


def get_prob_weight_initial_params(prob_weight: str):
    return PROB_WEIGHT_WRAPPERS[prob_weight].initial_params


class InvalidLotteryUtilityModelName(ValueError):
    pass
