import time
import numpy as np
import pandas as pd

from recommendation_data_toolbox.mle import estimate_max_lik_params
from .utils import (
    get_fullpath_to_datafile,
    get_probs_from_encodings,
    get_vals_from_encodings,
)


def estimate_params_across_df(
    df: pd.DataFrame, model: str, uf_model: str, is_with_constraints: bool
):
    a_ocs = get_vals_from_encodings(df["aValues"])
    a_probs = get_probs_from_encodings(df["aProbs"])
    b_ocs = get_vals_from_encodings(df["bValues"])
    b_probs = get_probs_from_encodings(df["bProbs"])
    observed_data = np.array(df["Risk"], dtype=bool)

    if uf_model == "power_nonneg":
        mask = np.all(a_ocs >= 0, axis=1) & np.all(b_ocs >= 0, axis=1)
        a_ocs, a_probs, b_ocs, b_probs, observed_data = (
            x[mask] for x in (a_ocs, a_probs, b_ocs, b_probs, observed_data)
        )

    start_s = time.time()
    res = estimate_max_lik_params(
        a_ocs=a_ocs,
        a_probs=a_probs,
        b_ocs=b_ocs,
        b_probs=b_probs,
        observed_data=observed_data,
        lottery_utility=model,
        outcome_utility=uf_model,
        is_with_constraints=is_with_constraints,
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


def estimate_params(
    experiment_number: int,
    model: str,
    is_neg_domain_included: bool,
    is_with_constraint: bool,
    is_per_subject: bool,
):
    """Perform mle on a utility model"""
    filename = f"IntermediateDataExperiment{experiment_number}.csv"
    df = pd.read_csv(get_fullpath_to_datafile(filename))
    args = dict(
        model=model,
        uf_model="power" if is_neg_domain_included else "power_nonneg",
        is_with_constraints=is_with_constraint,
    )
    return (
        df.groupby("SubjID").apply(estimate_params_across_df, **args)
        if is_per_subject
        else estimate_params_across_df(df, **args)
    )
