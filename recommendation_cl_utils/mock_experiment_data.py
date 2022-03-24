import pandas as pd
from sklearn.model_selection import train_test_split

from recommendation_cl_utils.constants import CPC15_DATASET_FILENAMES
from recommendation_cl_utils.utils import get_fullpath_to_datafile


def get_mock_data_from_df(df: pd.DataFrame):
    df = df[
        (df["Feedback"] == 0)
        & (df["Amb"] == 0)
        & (df["Manipulation"] == "Abstract")
    ]

    subj_ids = (
        df["Location"]
        + "_"
        + df["SubjID"].astype(str)
        + "_"
        + df["Trial"].astype(str)
    )
    return pd.DataFrame(
        dict(subj_id=subj_ids, problem_id=df["GameID"], decision=df["Risk"])
    )


def get_mock_data():
    df = get_mock_data_from_df(
        pd.read_csv(get_fullpath_to_datafile(CPC15_DATASET_FILENAMES[0]))
    )
    preexperiment_subj_ids, experiment_subj_ids = train_test_split(
        df["subj_id"].unique(), test_size=0.5, random_state=1234
    )
    return (
        pd.concat([df[df["subj_id"] == id] for id in preexperiment_subj_ids]),
        pd.concat([df[df["subj_id"] == id] for id in experiment_subj_ids]),
    )
