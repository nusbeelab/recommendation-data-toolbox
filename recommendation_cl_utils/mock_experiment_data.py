import pandas as pd
from sklearn.model_selection import train_test_split

from recommendation_cl_utils.constants import (
    CPC15_DATASET_FILENAMES,
    CPC15_LOTTERY_PAIR_HEADERS,
)
from recommendation_cl_utils.utils import get_fullpath_to_datafile


def get_mock_data_from_df(df: pd.DataFrame):
    df = df[
        (df["Feedback"] == 0)
        & (df["Amb"] == 0)
        & (df["Manipulation"] == "Abstract")
    ]

    df["SubjID"] = (
        df["Location"]
        + "_"
        + df["SubjID"].astype(str)
        + "_"
        + df["Trial"].astype(str)
    )

    lot_pairs = df[CPC15_LOTTERY_PAIR_HEADERS].drop_duplicates()
    if len(lot_pairs) != 25:
        raise ValueError

    return df, lot_pairs


def get_mock_data():
    df, lot_pair = get_mock_data_from_df(
        pd.read_csv(get_fullpath_to_datafile(CPC15_DATASET_FILENAMES[0]))
    )
    preexperiment_subj_ids, experiment_subj_ids = train_test_split(
        df["SubjID"].unique(), test_size=0.5, random_state=1234
    )
    return (
        pd.concat([df[df["SubjID"] == id] for id in preexperiment_subj_ids]),
        pd.concat([df[df["SubjID"] == id] for id in experiment_subj_ids]),
        lot_pair,
    )
