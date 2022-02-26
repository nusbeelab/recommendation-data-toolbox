import os
import numpy as np
import pandas as pd
from math import comb
import numpy.typing as npt

from param_estimation import CWD


def simplify_lottery(
    values: npt.NDArray[np.int_], probs: npt.NDArray[np.float64]
):
    mask = probs > 0
    return values[mask], probs[mask]


def get_probs_hb_lottery_skew(lot_num: int):
    return np.fromiter(
        (
            np.power(1 / 2, i) if i < lot_num else np.power(1 / 2, i - 1)
            for i in range(1, lot_num + 1)
        ),
        np.float64,
    )


def get_unnormalized_hb_lottery(
    hb: int, p_hb: np.float64, lot_num: int, lot_shape: str
):
    if lot_num < 1:
        raise ValueError("LotNum must be a positive integer.")

    values, probs = None, None

    if lot_num == 1:
        values, probs = [hb], [p_hb]
    elif lot_shape == "Symm":
        if lot_num % 2 == 0:
            raise ValueError(
                'LotNum must be an odd integer when LotShape is "Symm".'
            )
        k = lot_num - 1
        high_val = int(k / 2)
        low_val = -high_val
        values = hb + np.fromiter(range(low_val, high_val + 1), int)
        probs = (
            np.fromiter((comb(k, i) for i in range(lot_num)), int)
            * np.power(1 / 2, k, dtype=np.float64)
            * p_hb
        )
    elif lot_shape == "R-skew":
        c = -lot_num - 1
        values = (
            hb
            + c
            + np.fromiter((np.power(2, i) for i in range(1, lot_num + 1)), int)
        )
        probs = get_probs_hb_lottery_skew(lot_num) * p_hb
    elif lot_shape == "L-skew":
        c = lot_num + 1
        values = (
            hb
            + c
            - np.fromiter((np.power(2, i) for i in range(1, lot_num + 1)), int)
        )
        probs = get_probs_hb_lottery_skew(lot_num) * p_hb
    else:
        raise ValueError(
            'LotShape must be either "Symm", "L-skew", or "R-skew"'
        )
    return values, probs


def get_a_lottery_from_row(row: pd.Series):
    values = np.array([row["Ha"], row["La"]])
    probs = np.array([row["pHa"], 1 - row["pHa"]])
    return pd.Series(
        simplify_lottery(values, probs), index=["aValues", "aProbs"]
    )


def get_b_lottery_from_row(row: pd.Series):
    hb_values, hb_probs = get_unnormalized_hb_lottery(
        hb=row["Hb"],
        p_hb=row["pHb"],
        lot_num=row["LotNum"],
        lot_shape=row["LotShape"],
    )
    values = np.append(hb_values, row["Lb"])
    probs = np.append(hb_probs, 1 - row["pHb"])
    return pd.Series(
        simplify_lottery(values, probs), index=["bValues", "bProbs"]
    )


def get_immediate_data_from_csv(filepath: str):
    df = pd.read_csv(filepath)
    df = df[(df["Trial"] == 1) & (df["Amb"] == 0)]
    a_lotteries = df.apply(get_a_lottery_from_row, axis=1, result_type="expand")
    b_lotteries = df.apply(get_b_lottery_from_row, axis=1, result_type="expand")
    assert len(df) == len(a_lotteries)
    assert len(df) == len(b_lotteries)
    return pd.concat([df, a_lotteries, b_lotteries], axis=1)


def get_intermediate_data():
    filenames = [
        "RawDataExperiment1sorted.csv",
        "RawDataExperiment2sorted.csv",
        "RawDataExperiment3.csv",
    ]
    filepaths = [os.path.join(CWD, "data", filename) for filename in filenames]
    dfs = [get_immediate_data_from_csv(filepath) for filepath in filepaths]
    return pd.concat(dfs)
