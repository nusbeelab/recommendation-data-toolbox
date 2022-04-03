import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from recommendation_data_toolbox.lottery import get_problem_manager

from recommendation_cl_utils.utils import get_fullpath_to_datafile
from recommendation_cl_utils.rec_benchmarking.common import get_rating_matrix_df
from recommendation_cl_utils.rec_benchmarking.cf import (
    CF_RECOMMENDERS,
    benchmark_cf_model_per_fold,
)
from recommendation_cl_utils.rec_benchmarking.content_based import (
    CONTENT_BASED_RECOMMENDERS,
    benchmark_content_based_model_per_fold,
)


def benchmark_model(model: str, dataset: str):
    if model in CF_RECOMMENDERS:
        experiment_filename = f"MockExperimentData_{dataset}.csv"
        preexperiment_filename = f"MockPreexperimentData_{dataset}.csv"

        experiment_data = pd.read_csv(
            get_fullpath_to_datafile(experiment_filename)
        )

        experiment_rating_matrix_df = get_rating_matrix_df(experiment_data)
        problem_ids = np.arange(experiment_rating_matrix_df.shape[-1])

        if dataset == "CPC15":
            kf = KFold(n_splits=5, random_state=1234, shuffle=True)
            df = pd.concat(
                [
                    benchmark_cf_model_per_fold(
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
                    "accuracy": df["accuracy"][
                        df["subj_id"] == "overall"
                    ].mean(),
                    "feature_importances": np.nan,
                },
                index=["overall"],
            )
            return pd.concat([overall_acc_df, df]).dropna(axis=1, how="all")
        elif dataset == "preexperiment":
            return benchmark_cf_model_per_fold(
                experiment_rating_matrix_df=experiment_rating_matrix_df,
                fold_num=0,
                train_problem_ids=problem_ids[:60],
                test_problem_ids=problem_ids[60:],
                model=model,
                preexperiment_filename=preexperiment_filename,
            ).dropna(axis=1, how="all")
    elif model in CONTENT_BASED_RECOMMENDERS:
        rating_data_filename = f"Data_{dataset}.csv"
        data = pd.read_csv(get_fullpath_to_datafile(rating_data_filename))
        rating_matrix_df = get_rating_matrix_df(data)
        problem_ids = np.arange(rating_matrix_df.shape[-1])

        problems_filename = f"Problems_{'RecProj' if dataset in ['preexperiment', 'experiment'] else dataset}.csv"
        problem_manager = get_problem_manager(
            pd.read_csv(get_fullpath_to_datafile(problems_filename))
        )
        if dataset == "CPC15":
            kf = KFold(n_splits=5, random_state=1234, shuffle=True)
            df = pd.concat(
                [
                    benchmark_content_based_model_per_fold(
                        rating_matrix_df=rating_matrix_df,
                        fold_num=fold_num,
                        train_problem_ids=problem_ids[train_idx],
                        test_problem_ids=problem_ids[test_idx],
                        model=model,
                        problem_manager=problem_manager,
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
                    "accuracy": df["accuracy"][
                        df["subj_id"] == "overall"
                    ].mean(),
                    "feature_importances": np.nan,
                },
                index=["overall"],
            )
            return pd.concat([overall_acc_df, df]).dropna(axis=1, how="all")
        elif dataset == "preexperiment":
            return benchmark_content_based_model_per_fold(
                rating_matrix_df=rating_matrix_df,
                fold_num=0,
                train_problem_ids=problem_ids[:60],
                test_problem_ids=problem_ids[60:],
                model=model,
                problem_manager=problem_manager,
            ).dropna(axis=1, how="all")
