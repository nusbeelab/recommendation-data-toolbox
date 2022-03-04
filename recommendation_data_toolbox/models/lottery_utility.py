import numpy as np
import numpy.typing as npt

from recommendation_data_toolbox.models.outcome_utility import (
    OutcomeUtilityModel,
    get_outcome_utility_model,
)
from recommendation_data_toolbox.models.prob_weight import (
    ProbWeightModel,
    get_prob_weight_model,
)


class LotteryUtilityModel:
    def __init__(
        self,
        outcome_utility_model: OutcomeUtilityModel,
        prob_weight_model: ProbWeightModel,
    ):
        self.outcome_utility_model = outcome_utility_model
        self.prob_weight_model = prob_weight_model

        outcome_utility_param_num = len(
            self.outcome_utility_model.initial_params
        )

        def lottery_utility_func(
            params: tuple,
            outcomes: npt.NDArray[np.int_],
            probs: npt.NDArray[np.float64],
        ):
            utility_of_outcomes = (
                self.outcome_utility_model.outcome_utility_func(
                    params[:outcome_utility_param_num], outcomes
                )
            )
            prob_weights = self.prob_weight_model.prob_weight_func(
                params[outcome_utility_param_num:], probs
            )
            return np.sum(
                np.multiply(prob_weights, utility_of_outcomes), axis=-1
            )

        self.lottery_utility_func = lottery_utility_func
        self.bounds = (
            self.outcome_utility_model.bounds + self.prob_weight_model.bounds
        )
        self.inital_params = (
            self.outcome_utility_model.initial_params
            + self.prob_weight_model.initial_params
        )


def get_lottery_utility_model(
    lottery_utility_model_name: str,
    outcome_utility_model_name: str = "power_nonneg",
):
    return LotteryUtilityModel(
        get_outcome_utility_model(outcome_utility_model_name),
        get_prob_weight_model(lottery_utility_model_name),
    )
