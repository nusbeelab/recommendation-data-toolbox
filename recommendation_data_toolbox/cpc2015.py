from math import comb
from typing import Optional
import numpy as np

from recommendation_data_toolbox.lottery import simplify_lottery


def get_skewed_lottery_probs(lot_num: int):
    """Generates the probabilities for a lottery with a skewed shape.
    Descriptively, the lottery's distribution is a truncated geometric
    distribution with the parameter 1/2 with the last term's probability
    adjusted up such that the distribution is well-defined.
    """
    return np.fromiter(
        (
            np.power(1 / 2, i) if i < lot_num else np.power(1 / 2, i - 1)
            for i in range(1, lot_num + 1)
        ),
        np.float64,
    )


def unpack_lottery_distribution(
    high_val: int,
    high_prob: float,
    low_val: int,
    lot_num: Optional[int] = None,
    lot_shape: Optional[str] = None,
):
    """Obtains the probability distribution of a lottery, given its descriptors.

    Parameters
    ----------
    high_val : int
        The payoff of the high objective consequence, or the expected value of the high
        objective consequence if the option is a multi-objective consequence problem i.e.
        it includes more than two possible objective consequence.
    high_prob : float
        The probability of getting the high objective consequence.
    low_val : int, optional
        The payoff of the low objective consequence. Can be None if high_prob is 1.
    lot_num : str, optional
        The number of the possible high objective consequence if the option is a
        multi-objective consequence problem.
    lot_shape : str, optional
        The shape of the high objective consequence distribution.

    Returns
    -------
    objective_consequence : numpy array of int
    probs : numpy array of float64

    Raises
    ------
    ValueError
    """
    values, probs = None, None
    if lot_num is None or lot_num == 1:
        values = np.array([high_val])
        probs = np.array([high_prob])
    elif lot_num < 1:
        raise ValueError("LotNum must be a positive integer.")
    elif lot_shape == "Symm":
        if lot_num % 2 == 0:
            raise ValueError(
                'LotNum must be an odd integer when LotShape is "Symm".'
            )
        k = lot_num - 1
        upper = int(k / 2)
        lower = -upper
        values = high_val + np.fromiter(range(lower, upper + 1), int)
        probs = (
            np.fromiter((comb(k, i) for i in range(lot_num)), int)
            * np.power(1 / 2, k, dtype=np.float64)
            * high_prob
        )
    elif lot_shape == "R-skew":
        c = -lot_num - 1
        values = (
            high_val
            + c
            + np.fromiter((np.power(2, i) for i in range(1, lot_num + 1)), int)
        )
        probs = get_skewed_lottery_probs(lot_num) * high_prob
    elif lot_shape == "L-skew":
        c = lot_num + 1
        values = (
            high_val
            + c
            - np.fromiter((np.power(2, i) for i in range(1, lot_num + 1)), int)
        )
        probs = get_skewed_lottery_probs(lot_num) * high_prob
    else:
        raise ValueError(
            'LotShape must be either "Symm", "L-skew", or "R-skew"'
        )
    values = np.append(values, [low_val])
    probs = np.append(probs, [1 - high_prob])
    return simplify_lottery(values, probs)
