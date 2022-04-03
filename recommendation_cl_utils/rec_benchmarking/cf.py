import numpy as np
import pandas as pd

from typing import Dict, Literal, Optional, Type
import numpy.typing as npt

from recommendation_data_toolbox.rec.cf import CfRecommender
from recommendation_data_toolbox.rec.cf.model_based import (
    CfDecisionTreeRecommender,
    LatentFactorRecommender,
    CfNaiveBayesRecommender,
)
from recommendation_data_toolbox.rec.cf.neighborhood_based import (
    IbcfRecommender,
    UbcfRecommender,
)

from recommendation_cl_utils.utils import get_accuracy, get_fullpath_to_datafile
from recommendation_cl_utils.rec_benchmarking.common import get_rating_matrix_df


CF_RECOMMENDER_CLASSES: Dict[str, Type[CfRecommender]] = {
    "ubcf": UbcfRecommender,
    "ibcf": IbcfRecommender,
    "cf_decision_tree": CfDecisionTreeRecommender,
    "cf_naive_bayes": CfNaiveBayesRecommender,
    "latent_factor": LatentFactorRecommender,
}

CF_RECOMMENDERS = CF_RECOMMENDER_CLASSES.keys()


def get_cf_preds_per_subj(
    rating_matrix,
    subj_problem_ids: npt.NDArray,
    subj_decisions: npt.NDArray,
    subj_test_problem_ids: npt.NDArray,
    model: Literal[
        "ubcf", "ibcf", "cf_decision_tree", "cf_naive_bayes", "latent_factor"
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


def benchmark_cf_model_per_fold(
    experiment_rating_matrix_df: pd.DataFrame,
    fold_num: int,
    train_problem_ids: npt.NDArray[np.int_],
    test_problem_ids: npt.NDArray[np.int_],
    model: str,
    preexperiment_filename: str,
):
    subj_ids = experiment_rating_matrix_df.index.tolist()
    experiment_rating_matrix = experiment_rating_matrix_df.values
    train_decisions = experiment_rating_matrix[:, train_problem_ids]
    test_decisions = experiment_rating_matrix[:, test_problem_ids]

    preds = get_cf_preds_all_subjs(
        train_problem_ids,
        train_decisions,
        test_problem_ids,
        model,
        preexperiment_filename,
    )
    data = {
        "fold_num": fold_num,
        "subj_id": subj_ids,
        "train_problem_ids": [tuple(train_problem_ids)] * len(subj_ids),
        "train_decisions": [tuple(x) for x in train_decisions],
        "test_problem_ids": [tuple(test_problem_ids)] * len(subj_ids),
        "actual_decisions": [tuple(x) for x in test_decisions],
        "predicted_decisions": [tuple(x) for x in preds],
        "accuracy": get_accuracy(test_decisions, preds),
    }
    df = pd.DataFrame(
        data, index=[f"{fold_num}_{subj_id}" for subj_id in subj_ids]
    )
    overall_acc_df = pd.DataFrame(
        {
            "fold_num": fold_num,
            "subj_id": "overall",
            "train_problem_ids": np.nan,
            "train_decisions": np.nan,
            "test_problem_ids": np.nan,
            "actual_decisions": np.nan,
            "predicted_decisions": np.nan,
            "accuracy": get_accuracy(test_decisions.flatten(), preds.flatten()),
        },
        index=["f{fold_num}_overall"],
    )
    return pd.concat([overall_acc_df, df])
