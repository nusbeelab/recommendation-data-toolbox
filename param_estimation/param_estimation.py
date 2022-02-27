import os
import numpy as np
import pandas as pd
from typing import Any
import numpy.typing as npt
from param_estimation import CWD

from recommendation_data_toolbox.mle import UtilityFunc, mle_estimate
from param_estimation.utils import (
    get_probs_from_encodings,
    get_vals_from_encodings,
)
from recommendation_data_toolbox.utility_functions import (
    crra_utility_fn_1param,
    crra_utility_fn_3params,
)


def estimate_params_across_df(
    df: pd.DataFrame, x0: npt.NDArray[(Any,)], utility_fn: UtilityFunc
):
    a_values = get_vals_from_encodings(df["aValues"])
    a_probs = get_probs_from_encodings(df["aProbs"])
    b_values = get_vals_from_encodings(df["bValues"])
    b_probs = get_probs_from_encodings(df["bProbs"])
    observed_data = np.array(df["Risk"], dtype=bool)

    if utility_fn == crra_utility_fn_1param:
        mask = np.all(a_values >= 0, axis=1) & np.all(b_values >= 0, axis=1)
        a_values, a_probs, b_values, b_probs, data = (
            x[mask] for x in (a_values, a_probs, b_values, b_probs, data)
        )

    res = mle_estimate(
        x0=x0,
        utility_fn=utility_fn,
        a_values=a_values,
        a_probs=a_probs,
        b_values=b_values,
        b_probs=b_probs,
        observed_data=observed_data,
    )
    return pd.Series(
        [res.success, res.fun, res.x],
        index=["isSuccess", "minNegLogLik", "optimalParams"],
    )


UTILITY_MODELS = {
    "crra1param": dict(utility_fn=crra_utility_fn_1param, params_num=1),
    "crra3params": dict(utility_fn=crra_utility_fn_3params, params_num=3),
}


def estimate_params_by_subjects(filename: str, utility_model: str):
    """Perform mle on a utility model"""
    filepath = os.path.join(CWD, "data", filename)
    df = pd.read_csv(filepath)
    return df.groupby("SubjID").apply(
        estimate_params_across_df,
        x0=np.ones((UTILITY_MODELS[utility_model]["params_num"],)),
        utility_fn=UTILITY_MODELS[utility_model]["utility_fn"],
    )
