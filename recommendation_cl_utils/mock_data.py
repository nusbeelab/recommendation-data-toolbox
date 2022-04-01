from typing import Literal
import pandas as pd
from sklearn.model_selection import train_test_split

from recommendation_cl_utils.constants import CPC15_DATASET_FILENAMES
from recommendation_cl_utils.utils import get_fullpath_to_datafile


def get_mock_data(dataset: Literal["CPC15", "preexperiment"]):
    if dataset == "CPC15":
        df = pd.read_csv(get_fullpath_to_datafile(CPC15_DATASET_FILENAMES[0]))
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
    elif dataset == "preexperiment":
        df_intro = pd.read_csv(
            get_fullpath_to_datafile("preexperiment_intro_2022-03-24.csv")
        )
        subj_ids = df_intro["participant.code"][
            df_intro["participant._current_page_name"] == "Finish"
        ].to_list()
        df = pd.read_csv(
            get_fullpath_to_datafile(
                "preexperiment_binarychoicequestions_results.csv"
            )
        )
        df = df[df["participant_code"].isin(subj_ids)]
        return df.rename(
            dict(participant_code="subj_id", response="decision"), axis=1
        )[["subj_id", "problem_id", "decision"]]


def get_mock_split_data(dataset: Literal["CPC15", "preexperiment"]):
    df = get_mock_data(dataset)
    if dataset == "CPC15":
        preexperiment_subj_ids, experiment_subj_ids = train_test_split(
            df["subj_id"].unique(), test_size=0.5, random_state=1234
        )
        return pd.concat(
            [df[df["subj_id"] == id] for id in preexperiment_subj_ids]
        ), pd.concat([df[df["subj_id"] == id] for id in experiment_subj_ids])
    elif dataset == "preexperiment":
        subj_ids = df["subj_id"].unique().tolist()
        preexperiment_subj_ids = subj_ids[:350]
        experiment_subj_ids = subj_ids[350:]

        preexperiment_df = df[df["subj_id"].isin(preexperiment_subj_ids)]
        experiment_df = df[df["subj_id"].isin(experiment_subj_ids)]

        return preexperiment_df, experiment_df
