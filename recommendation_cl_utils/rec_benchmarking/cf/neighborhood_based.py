from typing import Dict, Literal, Type
import numpy as np
import pandas as pd
import numpy.typing as npt
from recommendation_cl_utils.constants import CPC15_LOTTERY_PAIR_HEADERS
from recommendation_cl_utils.utils import get_fullpath_to_datafile

from recommendation_data_toolbox.rec.cf.neighborhood_based import (
    IbcfRecommender,
    NbcfRecommender,
    UbcfRecommender,
)


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
    return [
        decision for _, decision in sorted(zip(lottery_pair_ids, decisions))
    ]


def get_rating_matrix(lot_pair_to_id_dict: Dict[tuple, int]):
    preexperiment_data = pd.read_csv(
        get_fullpath_to_datafile("MockPreexperimentData.csv")
    )
    preexperiment_data["lottery_pair"] = preexperiment_data[
        CPC15_LOTTERY_PAIR_HEADERS
    ].apply(tuple, axis=1)
    return np.array(
        preexperiment_data.groupby("SubjID")
        .apply(
            func=convert_subj_data_to_rating_vector,
            lot_pair_to_id_dict=lot_pair_to_id_dict,
        )
        .to_list()
    )


NBCF_MODEL_CLASSES: Dict[str, Type[NbcfRecommender]] = {
    "ubcf": UbcfRecommender,
    "ibcf": IbcfRecommender,
}


def get_nbcf_preds_per_subj(
    rating_matrix: npt.NDArray,
    subj_lot_pair_ids: npt.NDArray,
    subj_decisions: npt.NDArray,
    subj_test_lot_pair_ids: npt.NDArray,
    model: Literal["ubcf", "ibcf"],
):
    recommender = NBCF_MODEL_CLASSES[model](
        rating_matrix=rating_matrix,
        subj_lot_pair_ids=subj_lot_pair_ids,
        subj_decisions=subj_decisions,
    )
    return [
        recommender.rec(lot_pair_id) for lot_pair_id in subj_test_lot_pair_ids
    ]


def get_nbcf_preds(
    train_lot_pair_ids: npt.NDArray,
    train_decisions: npt.NDArray,
    test_lot_pair_ids: npt.NDArray,
    lot_pair_to_id_dict,
    model: Literal["ubcf", "ibcf"],
):
    rating_matrix = get_rating_matrix(lot_pair_to_id_dict)
    return np.array(
        [
            get_nbcf_preds_per_subj(
                rating_matrix,
                subj_lot_pair_ids,
                subj_decisions,
                subj_test_lot_pair_ids,
                model,
            )
            for subj_lot_pair_ids, subj_decisions, subj_test_lot_pair_ids in zip(
                train_lot_pair_ids, train_decisions, test_lot_pair_ids
            )
        ]
    )


def get_ubcf_preds(
    train_lot_pair_ids: npt.NDArray,
    train_decisions: npt.NDArray,
    test_lot_pair_ids: npt.NDArray,
    lot_pair_to_id_dict,
):
    return get_nbcf_preds(
        train_lot_pair_ids,
        train_decisions,
        test_lot_pair_ids,
        lot_pair_to_id_dict,
        "ubcf",
    )


def get_ibcf_preds(
    train_lot_pair_ids: npt.NDArray,
    train_decisions: npt.NDArray,
    test_lot_pair_ids: npt.NDArray,
    lot_pair_to_id_dict,
):
    return get_nbcf_preds(
        train_lot_pair_ids,
        train_decisions,
        test_lot_pair_ids,
        lot_pair_to_id_dict,
        "ibcf",
    )
