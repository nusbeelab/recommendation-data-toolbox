import numpy as np
import pandas as pd
import numpy.typing as npt

from recommendation_cl_utils.constants import CPC15_LOTTERY_PAIR_HEADERS
from recommendation_cl_utils.rec_benchmarking.cf.neighborhood_based import (
    get_ibcf_preds,
    get_ubcf_preds,
)
from recommendation_cl_utils.utils import get_accuracy, get_fullpath_to_datafile


def get_lot_pair_to_id_dict():
    lottery_pairs = (
        pd.read_csv(get_fullpath_to_datafile("MockBinaryChoices.csv"))
        .apply(tuple, axis=1)
        .to_dict()
    )
    return {v: k for k, v in lottery_pairs.items()}


def agg_data_per_subj(df: pd.DataFrame):
    return pd.Series(
        [df.index[0], df["lot_pair_id"].tolist(), df["Risk"].tolist()],
        index=["SubjID", "lot_pair_ids", "decisions"],
    )


def shuffle_columns(arr: npt.NDArray):
    return arr[:, np.random.RandomState(seed=1234).permutation(arr.shape[1])]


def get_benchmark_data(experiment_data: pd.DataFrame, lot_pair_to_id_dict):
    df = experiment_data.copy()
    df["lot_pair_id"] = (
        experiment_data[CPC15_LOTTERY_PAIR_HEADERS]
        .apply(tuple, axis=1)
        .apply(lambda x: lot_pair_to_id_dict[x])
    )
    agg_data_df = df.groupby("SubjID").apply(agg_data_per_subj)
    subj_ids, lot_pair_ids, decisions = agg_data_df.T.values.tolist()
    lot_pair_ids, decisions = (
        shuffle_columns(np.array(x)) for x in (lot_pair_ids, decisions)
    )

    train_lot_pair_ids = lot_pair_ids[:, :20]
    train_decisions = decisions[:, :20]

    test_lot_pair_ids = lot_pair_ids[:, 20:]
    test_decisions = decisions[:, 20:]

    return (
        subj_ids,
        train_lot_pair_ids,
        train_decisions,
        test_lot_pair_ids,
        test_decisions,
    )


PREDS_GETTERS = {"ubcf": get_ubcf_preds, "ibcf": get_ibcf_preds}


def benchmark_model(model: str):
    experiment_data = pd.read_csv(
        get_fullpath_to_datafile("MockExperimentData.csv")
    )
    lot_pair_to_id_dict = get_lot_pair_to_id_dict()
    (
        subj_ids,
        train_lot_pair_ids,
        train_decisions,
        test_lot_pair_ids,
        test_decisions,
    ) = get_benchmark_data(experiment_data, lot_pair_to_id_dict)
    preds = PREDS_GETTERS[model](
        train_lot_pair_ids,
        train_decisions,
        test_lot_pair_ids,
        lot_pair_to_id_dict,
    )
    headers_out = [
        "SubjID",
        "TrainProblemIDs",
        "TrainRisks",
        "TestProblemIDs",
        "ActualRisks",
        "PredictedRisks",
        "Accuracy",
    ]
    data_out = [
        subj_ids,
        train_lot_pair_ids.tolist(),
        train_decisions.tolist(),
        test_lot_pair_ids.tolist(),
        test_decisions.tolist(),
        preds.tolist(),
        get_accuracy(test_decisions, preds).tolist(),
    ]

    df_out_data = {k: v for k, v in zip(headers_out, data_out)}
    df_out = pd.DataFrame(df_out_data)

    overall_acc_df = pd.DataFrame(
        [
            [
                "Overall",
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                get_accuracy(test_decisions.flatten(), preds.flatten()),
            ]
        ],
        columns=headers_out,
    )
    df_out = pd.concat([overall_acc_df, df_out])
    df_out.to_csv(
        get_fullpath_to_datafile(f"benchmark_{model}.csv"), index=False
    )
