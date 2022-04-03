import numpy as np
import pandas as pd

from typing import Dict, Type
import numpy.typing as npt

from recommendation_data_toolbox.lottery import ProblemManager

from recommendation_data_toolbox.rec.content_based import (
    ContentBasedDecisionTreeRecommender,
    ContentBasedGradientBoostingRecommender,
    ContentBasedNaiveBayesRecommender,
    ContentBasedRandomForestRecommender,
    ContentBasedRecommender,
)

from recommendation_cl_utils.utils import get_accuracy

CONTENT_BASED_RECOMMENDER_CLASSES: Dict[str, Type[ContentBasedRecommender]] = {
    "content_based_decision_tree": ContentBasedDecisionTreeRecommender,
    "content_based_random_forest": ContentBasedRandomForestRecommender,
    "content_based_gradient_boosting": ContentBasedGradientBoostingRecommender,
    # "content_based_xgboost": ContentBasedXgboostRecommender,
    "content_based_naive_bayes": ContentBasedNaiveBayesRecommender,
}

CONTENT_BASED_RECOMMENDERS = CONTENT_BASED_RECOMMENDER_CLASSES.keys()


def get_content_based_preds_and_feature_importances_per_subj(
    problem_manager: ProblemManager,
    subj_problem_ids: npt.NDArray,
    subj_decisions: npt.NDArray,
    subj_test_problem_ids: npt.NDArray,
    subj_test_decisions: npt.NDArray,
    model: str,
):
    recommender = CONTENT_BASED_RECOMMENDER_CLASSES[model](
        problem_manager=problem_manager,
        subj_problem_ids=subj_problem_ids,
        subj_decisions=subj_decisions,
    )
    preds = recommender.rec(subj_test_problem_ids)
    feature_importances = recommender.get_feature_importance(
        test_problem_ids=subj_test_problem_ids,
        test_decisions=subj_test_decisions,
    )
    return preds, feature_importances


def get_content_based_preds_and_feature_importances_all_subjs(
    train_problem_ids: npt.NDArray[np.int_],
    train_decisions: npt.NDArray[np.int_],
    test_problem_ids: npt.NDArray[np.int_],
    test_decisions: npt.NDArray[np.int_],
    model: str,
    problem_manager: ProblemManager,
):
    preds_and_feature_importances = [
        get_content_based_preds_and_feature_importances_per_subj(
            problem_manager,
            train_problem_ids,
            subj_decisions,
            test_problem_ids,
            subj_test_decisions,
            model,
        )
        for subj_decisions, subj_test_decisions in zip(
            train_decisions, test_decisions
        )
    ]
    preds = np.array([x[0] for x in preds_and_feature_importances])
    feature_importances = np.array(
        [x[1] for x in preds_and_feature_importances]
    )
    return preds, feature_importances


def benchmark_content_based_model_per_fold(
    rating_matrix_df: pd.DataFrame,
    fold_num: int,
    train_problem_ids: npt.NDArray[np.int_],
    test_problem_ids: npt.NDArray[np.int_],
    model: str,
    problem_manager: ProblemManager,
):
    subj_ids = rating_matrix_df.index.tolist()
    rating_matrix = rating_matrix_df.values
    train_decisions = rating_matrix[:, train_problem_ids]
    test_decisions = rating_matrix[:, test_problem_ids]
    (
        preds,
        feature_importances,
    ) = get_content_based_preds_and_feature_importances_all_subjs(
        train_problem_ids,
        train_decisions,
        test_problem_ids,
        test_decisions,
        model,
        problem_manager,
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
        "feature_importances": [tuple(x) for x in feature_importances],
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
            "feature_importances": np.nan,
        },
        index=["f{fold_num}_overall"],
    )
    return pd.concat([overall_acc_df, df])
