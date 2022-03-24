import numpy as np
import pandas as pd

from typing import Dict, Literal, Type
import numpy.typing as npt

from recommendation_data_toolbox.rec.cf import CfRecommender
from recommendation_data_toolbox.rec.cf.model_based import (
    DecisionTreeRecommender,
    LatentFactorRecommender,
    NaiveBayesRecommender,
)
from recommendation_data_toolbox.rec.cf.neighborhood_based import (
    IbcfRecommender,
    UbcfRecommender,
)

from recommendation_cl_utils.utils import get_fullpath_to_datafile
from recommendation_cl_utils.rec_benchmarking.common import get_rating_matrix_df


CF_RECOMMENDER_CLASSES: Dict[str, Type[CfRecommender]] = {
    "ubcf": UbcfRecommender,
    "ibcf": IbcfRecommender,
    "decision_tree": DecisionTreeRecommender,
    "naive_bayes": NaiveBayesRecommender,
    "latent_factor": LatentFactorRecommender,
}


def get_nbcf_preds_per_subj(
    rating_matrix,
    subj_lot_pair_ids: npt.NDArray,
    subj_decisions: npt.NDArray,
    subj_test_lot_pair_ids: npt.NDArray,
    model: Literal[
        "ubcf", "ibcf", "decision_tree", "naive_bayes", "latent_factor"
    ],
):

    recommender = CF_RECOMMENDER_CLASSES[model](
        rating_matrix=rating_matrix,
        subj_lot_pair_ids=subj_lot_pair_ids,
        subj_decisions=subj_decisions,
    )
    return np.array(
        [recommender.rec(lot_pair_id) for lot_pair_id in subj_test_lot_pair_ids]
    )


def get_nbcf_preds_all_subjs(
    lot_pair_to_id_dict,
    train_lot_pair_ids: npt.NDArray,
    train_decisions: npt.NDArray,
    test_lot_pair_ids: npt.NDArray,
    model: str,
):
    assert train_lot_pair_ids.shape == (20,)
    assert train_decisions.shape[1] == 20
    assert test_lot_pair_ids.shape == (5,)
    preexperiment_data = pd.read_csv(
        get_fullpath_to_datafile("MockPreexperimentData.csv")
    )
    rating_matrix = get_rating_matrix_df(
        preexperiment_data, lot_pair_to_id_dict
    ).values
    return np.array(
        [
            get_nbcf_preds_per_subj(
                rating_matrix,
                train_lot_pair_ids,
                subj_decisions,
                test_lot_pair_ids,
                model,
            )
            for subj_decisions in train_decisions
        ]
    )
