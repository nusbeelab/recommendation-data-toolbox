import os
import time
import numpy as np
import pandas as pd
from param_estimation import CWD

from recommendation_data_toolbox.mle import estimate_max_lik_params
from param_estimation.utils import (
    get_probs_from_encodings,
    get_vals_from_encodings,
)
from recommendation_data_toolbox.utility_functions import (
    UtilityFuncWrapper,
    crra1param_utility_fn,
    get_utility_func_wrapper,
)


def estimate_params_across_df(
    df: pd.DataFrame, utility_func_wrapper: UtilityFuncWrapper
):
    a_values = get_vals_from_encodings(df["aValues"])
    a_probs = get_probs_from_encodings(df["aProbs"])
    b_values = get_vals_from_encodings(df["bValues"])
    b_probs = get_probs_from_encodings(df["bProbs"])
    observed_data = np.array(df["Risk"], dtype=bool)

    if utility_func_wrapper.utility_fn is crra1param_utility_fn:
        mask = np.all(a_values >= 0, axis=1) & np.all(b_values >= 0, axis=1)
        a_values, a_probs, b_values, b_probs, observed_data = (
            x[mask]
            for x in (a_values, a_probs, b_values, b_probs, observed_data)
        )

    start_s = time.time()
    res = estimate_max_lik_params(
        utility_func_wrapper=utility_func_wrapper,
        a_values=a_values,
        a_probs=a_probs,
        b_values=b_values,
        b_probs=b_probs,
        observed_data=observed_data,
    )
    seconds_elapsed = time.time() - start_s
    return pd.Series(
        [
            res.success,
            res.fun,
            res.x,
            res.nit,
            seconds_elapsed,
        ],
        index=[
            "isSuccess",
            "minNegLogLik",
            "optimalParams",
            "iterNum",
            "secondsElapsed",
        ],
    )


def estimate_params_by_subjects(filename: str, utility_model: str):
    """Perform mle on a utility model"""
    filepath = os.path.join(CWD, "data", filename)
    df = pd.read_csv(filepath)
    return df.groupby("SubjID").apply(
        estimate_params_across_df,
        utility_func_wrapper=get_utility_func_wrapper(utility_model),
    )
