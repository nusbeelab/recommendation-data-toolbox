import pandas as pd
import numpy.typing as npt


def convert_row_to_lottery_pair(row: pd.Series):
    a_prob = row["Ha"]


def convert_to_rating_matrix(experiment_data: pd.DataFrame):
    lottery_pair_param_headers = [
        "Ha",
        "pHa",
        "La",
        "Hb",
        "pHb",
        "Lb",
        "LotShape",
        "LotNum",
    ]
    lottery_pair_params = experiment_data[
        lottery_pair_param_headers
    ].drop_duplicates()
    assert len(lottery_pair_params) == 30

    pass


def benchmark_ubcf(
    pre_experiment_data: npt.NDArray, experiment_data: npt.NDArray
):
    # lottery_pairs = pre_experiment_data[[""]]
    pass
