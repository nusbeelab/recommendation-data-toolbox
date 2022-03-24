import pandas as pd
from typing import Dict

from recommendation_cl_utils.constants import CPC15_LOTTERY_PAIR_HEADERS
from recommendation_cl_utils.utils import get_fullpath_to_datafile


def convert_subj_data_to_rating_vector(
    subj_experiment_data: pd.DataFrame,
    lot_pair_to_id_dict: Dict[tuple, int],
):
    if len(subj_experiment_data) != 25:
        raise ValueError(
            f"Each subject is expected to have 25 responses, but subject {subj_experiment_data['SubjID']} has {len(subj_experiment_data)} responses."
        )
    lottery_pair_ids = [
        lot_pair_to_id_dict[lot_pair]
        for lot_pair in subj_experiment_data["lottery_pair"]
    ]
    decisions = subj_experiment_data["Risk"].to_list()
    lottery_pair_ids, decisions = zip(*sorted(zip(lottery_pair_ids, decisions)))
    return pd.Series(
        decisions, index=lottery_pair_ids, name=subj_experiment_data.index[0]
    )


def get_rating_matrix_df(
    df: pd.DataFrame, lot_pair_to_id_dict: Dict[tuple, int]
):
    df = df.copy()
    df["lottery_pair"] = df[CPC15_LOTTERY_PAIR_HEADERS].apply(tuple, axis=1)
    return df.groupby("SubjID").apply(
        func=convert_subj_data_to_rating_vector,
        lot_pair_to_id_dict=lot_pair_to_id_dict,
    )


def get_lot_pair_to_id_dict():
    lottery_pairs = (
        pd.read_csv(get_fullpath_to_datafile("MockBinaryChoices.csv"))
        .apply(tuple, axis=1)
        .to_dict()
    )
    return {v: k for k, v in lottery_pairs.items()}
