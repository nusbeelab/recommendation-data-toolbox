from typing import Dict
import numpy as np
import pandas as pd
import numpy.typing as npt
from sklearn.model_selection import KFold

from recommendation_cl_utils.rec_benchmarking.cf.neighborhood_based import (
    get_ubcf_preds_all_subjs,
    get_ibcf_preds_all_subjs,
)
from recommendation_cl_utils.rec_benchmarking.common import (
    get_lot_pair_to_id_dict,
    get_rating_matrix_df,
)
from recommendation_cl_utils.utils import get_accuracy, get_fullpath_to_datafile


PREDS_GETTERS = {
    "ubcf": get_ubcf_preds_all_subjs,
    "ibcf": get_ibcf_preds_all_subjs,
}


def agg_data_per_subj(df: pd.DataFrame):
    lot_pair_ids, decisions = zip(
        *sorted(zip(df["lot_pair_id"].tolist(), df["Risk"].tolist()))
    )
    assert lot_pair_ids == list(range(25))
    return pd.Series(
        [df.index[0], lot_pair_ids, decisions],
        index=["SubjID", "lot_pair_ids", "decisions"],
    )


def benchmark_model_per_fold(
    experiment_rating_matrix_df: pd.DataFrame,
    lot_pair_to_id_dict: Dict[tuple, int],
    fold_num: int,
    train_lot_pair_ids: npt.NDArray[np.int_],
    test_lot_pair_ids: npt.NDArray[np.int_],
    model: str,
):
    subj_ids = experiment_rating_matrix_df.index.tolist()
    experiment_rating_matrix = experiment_rating_matrix_df.values
    train_decisions = experiment_rating_matrix[:, train_lot_pair_ids]
    test_decisions = experiment_rating_matrix[:, test_lot_pair_ids]

    preds = PREDS_GETTERS[model](
        lot_pair_to_id_dict,
        train_lot_pair_ids,
        train_decisions,
        test_lot_pair_ids,
    )
    data = {
        "FoldNum": fold_num,
        "SubjID": subj_ids,
        "TrainProblemIDs": [tuple(train_lot_pair_ids)] * len(subj_ids),
        "TrainRisks": [tuple(x) for x in train_decisions],
        "TestProblemIDs": [tuple(test_lot_pair_ids)] * len(subj_ids),
        "ActualRisks": [tuple(x) for x in test_decisions],
        "PredictedRisks": [tuple(x) for x in preds],
        "Accuracy": get_accuracy(test_decisions, preds),
    }
    df = pd.DataFrame(
        data, index=[f"{fold_num}_{subj_id}" for subj_id in subj_ids]
    )
    overall_acc_df = pd.DataFrame(
        {
            "FoldNum": fold_num,
            "SubjID": "overall",
            "TrainProblemIDs": np.nan,
            "TrainRisks": np.nan,
            "TestProblemIDs": np.nan,
            "ActualRisks": np.nan,
            "PredictedRisks": np.nan,
            "Accuracy": get_accuracy(test_decisions.flatten(), preds.flatten()),
        },
        index=["f{fold_num}_overall"],
    )
    return pd.concat([overall_acc_df, df])


def benchmark_model(model: str):
    experiment_data = pd.read_csv(
        get_fullpath_to_datafile("MockExperimentData.csv")
    )
    lot_pair_to_id_dict = get_lot_pair_to_id_dict()
    experiment_rating_matrix_df = get_rating_matrix_df(
        experiment_data, lot_pair_to_id_dict
    )
    lot_pair_ids = np.array(list(lot_pair_to_id_dict.values()))
    kf = KFold(n_splits=5, random_state=1234, shuffle=True)
    df = pd.concat(
        [
            benchmark_model_per_fold(
                experiment_rating_matrix_df=experiment_rating_matrix_df,
                lot_pair_to_id_dict=lot_pair_to_id_dict,
                fold_num=fold_num,
                train_lot_pair_ids=lot_pair_ids[train_idx],
                test_lot_pair_ids=lot_pair_ids[test_idx],
                model=model,
            )
            for fold_num, (train_idx, test_idx) in enumerate(
                kf.split(lot_pair_ids)
            )
        ]
    )
    overall_acc_df = pd.DataFrame(
        {
            "FoldNum": "overall",
            "SubjID": np.nan,
            "TrainProblemIDs": np.nan,
            "TrainRisks": np.nan,
            "TestProblemIDs": np.nan,
            "ActualRisks": np.nan,
            "PredictedRisks": np.nan,
            "Accuracy": df["Accuracy"][df["SubjID"] == "overall"].mean(),
        },
        index=["overall"],
    )
    df = pd.concat([overall_acc_df, df])
    df.to_csv(get_fullpath_to_datafile(f"benchmark_{model}.csv"), index=False)
