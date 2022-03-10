import os
import pandas as pd
from sklearn.model_selection import train_test_split

from recommendation_cl_utils import CWD
from recommendation_cl_utils.constants import CPC15_DATASET_FILENAMES

# def

# 30 questions per subject


def get_mock_data_from_df(df: pd.DataFrame, experiment_number):
    df = df[df["Feedback"] == 0]
    df["SubjID"] = (
        df["SubjID"].astype(str)
        + f"_{experiment_number}_"
        + df["Trial"].astype(str)
    )
    df = df.groupby("SubjID").filter(lambda x: len(x) == 30)
    df = df.drop("Trial", axis=1)
    return df


def get_mock_data():
    df = pd.concat(
        [
            get_mock_data_from_df(
                pd.read_csv(os.path.join(CWD, "data", filename)), idx + 1
            )
            for idx, filename in enumerate(CPC15_DATASET_FILENAMES)
        ]
    )
    preexperiment_subj_ids, experiment_subj_ids = train_test_split(
        df["SubjID"].drop_duplicates(), test_size=0.5, random_state=1234
    )
    return (
        df[df["SubjID"].isin(preexperiment_subj_ids)],
        df[df["SubjID"].isin(experiment_subj_ids)],
    )
