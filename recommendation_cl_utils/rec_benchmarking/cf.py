import numpy as np
import pandas as pd

from typing import Dict, Literal, Optional, Type
import numpy.typing as npt

from recommendation_data_toolbox.rec.cf import CfRecommender
from recommendation_data_toolbox.rec.cf.model_based import (
    DecisionTreeRecommender,
    LatentFactorRecommender,
    NaiveBayesRecommender,
)
from recommendation_data_toolbox.rec.cf.neighborhood_based import (
    IbcfRecommender,
    UbcfRecommender,
)

from recommendation_cl_utils.utils import get_fullpath_to_datafile
from recommendation_cl_utils.rec_benchmarking.common import get_rating_matrix_df


CF_RECOMMENDER_CLASSES: Dict[str, Type[CfRecommender]] = {
    "ubcf": UbcfRecommender,
    "ibcf": IbcfRecommender,
    "decision_tree": DecisionTreeRecommender,
    "naive_bayes": NaiveBayesRecommender,
    "latent_factor": LatentFactorRecommender,
}


def get_cf_preds_per_subj(
    rating_matrix,
    subj_problem_ids: npt.NDArray,
    subj_decisions: npt.NDArray,
    subj_test_problem_ids: npt.NDArray,
    model: Literal[
        "ubcf", "ibcf", "decision_tree", "naive_bayes", "latent_factor"
    ],
):

    recommender = CF_RECOMMENDER_CLASSES[model](
        rating_matrix=rating_matrix,
        subj_problem_ids=subj_problem_ids,
        subj_decisions=subj_decisions,
    )
    return recommender.rec(subj_test_problem_ids)


def get_cf_preds_all_subjs(
    train_problem_ids: npt.NDArray,
    train_decisions: npt.NDArray,
    test_problem_ids: npt.NDArray,
    model: str,
    preexperiment_filename: Optional[str],
):
    if preexperiment_filename == None:
        preexperiment_filename = "MockPreexperimentData.csv"
    preexperiment_data = pd.read_csv(
        get_fullpath_to_datafile(preexperiment_filename)
    )

    rating_matrix = get_rating_matrix_df(preexperiment_data).values

    return np.array(
        [
            get_cf_preds_per_subj(
                rating_matrix,
                train_problem_ids,
                subj_decisions,
                test_problem_ids,
                model,
            )
            for subj_decisions in train_decisions
        ]
    )
