import os
import pandas as pd

from param_estimation import CWD
from recommendation_data_toolbox.lottery import unpack_lottery_distribution


def get_a_lottery_from_row(row: pd.Series):
    return pd.Series(
        unpack_lottery_distribution(
            high_val=row["Ha"],
            high_prob=row["pHa"],
            low_val=row["La"],
        ),
        index=["aValues", "aProbs"],
    )


def get_b_lottery_from_row(row: pd.Series):
    return pd.Series(
        unpack_lottery_distribution(
            high_val=row["Hb"],
            high_prob=row["pHb"],
            low_val=row["Lb"],
            lot_num=row["LotNum"],
            lot_shape=row["LotShape"],
        ),
        index=["bValues", "bProbs"],
    )


def get_immediate_data_from_csv(filepath: str):
    df = pd.read_csv(filepath)
    df = df[(df["Trial"] == 1) & (df["Amb"] == 0)]
    a_lotteries = df.apply(get_a_lottery_from_row, axis=1, result_type="expand")
    b_lotteries = df.apply(get_b_lottery_from_row, axis=1, result_type="expand")
    assert len(df) == len(a_lotteries)
    assert len(df) == len(b_lotteries)
    return pd.concat([df, a_lotteries, b_lotteries], axis=1)


FILENAMES = [
    "RawDataExperiment1sorted.csv",
    "RawDataExperiment2sorted.csv",
    "RawDataExperiment3.csv",
]


def get_intermediate_data(experiment_number):
    filepath = os.path.join(CWD, "data", FILENAMES[experiment_number - 1])
    return get_immediate_data_from_csv(filepath)
