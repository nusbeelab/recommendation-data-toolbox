import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from typing import Dict, Optional
import numpy.typing as npt

from recommendation_cl_utils.utils import get_accuracy, get_fullpath_to_datafile
from recommendation_cl_utils.rec_benchmarking.common import get_rating_matrix_df
from recommendation_cl_utils.rec_benchmarking.cf import get_cf_preds_all_subjs


def get_preds_all_subjs(
    train_problem_ids,
    train_decisions,
    test_problem_ids,
    model,
    **kwargs,
):
    if model in [
        "ubcf",
        "ibcf",
        "decision_tree",
        "naive_bayes",
        "latent_factor",
    ]:
        return get_cf_preds_all_subjs(
            train_problem_ids,
            train_decisions,
            test_problem_ids,
            model,
            **kwargs,
        )
    else:
        raise ValueError


def benchmark_model_per_fold(
    experiment_rating_matrix_df: pd.DataFrame,
    fold_num: int,
    train_problem_ids: npt.NDArray[np.int_],
    test_problem_ids: npt.NDArray[np.int_],
    model: str,
    **kwargs,
):
    subj_ids = experiment_rating_matrix_df.index.tolist()
    experiment_rating_matrix = experiment_rating_matrix_df.values
    train_decisions = experiment_rating_matrix[:, train_problem_ids]
    test_decisions = experiment_rating_matrix[:, test_problem_ids]

    preds = get_preds_all_subjs(
        train_problem_ids,
        train_decisions,
        test_problem_ids,
        model,
        **kwargs,
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


def benchmark_model(
    model: str,
    experiment_filename: Optional[str] = None,
    preexperiment_filename: Optional[str] = None,
):
    if experiment_filename == None:
        experiment_filename = "MockExperimentData.csv"
    experiment_data = pd.read_csv(get_fullpath_to_datafile(experiment_filename))

    experiment_rating_matrix_df = get_rating_matrix_df(experiment_data)
    problem_ids = np.arange(experiment_rating_matrix_df.shape[-1])

    kf = KFold(n_splits=5, random_state=1234, shuffle=True)
    df = pd.concat(
        [
            benchmark_model_per_fold(
                experiment_rating_matrix_df=experiment_rating_matrix_df,
                fold_num=fold_num,
                train_problem_ids=problem_ids[train_idx],
                test_problem_ids=problem_ids[test_idx],
                model=model,
                preexperiment_filename=preexperiment_filename,
            )
            for fold_num, (train_idx, test_idx) in enumerate(
                kf.split(problem_ids)
            )
        ]
    )
    overall_acc_df = pd.DataFrame(
        {
            "fold_num": "overall",
            "subj_id": np.nan,
            "train_problem_ids": np.nan,
            "train_decisions": np.nan,
            "test_problem_ids": np.nan,
            "actual_decisions": np.nan,
            "predicted_decisions": np.nan,
            "accuracy": df["accuracy"][df["subj_id"] == "overall"].mean(),
        },
        index=["overall"],
    )
    df = pd.concat([overall_acc_df, df])
    df.to_csv(get_fullpath_to_datafile(f"benchmark_{model}.csv"), index=False)
