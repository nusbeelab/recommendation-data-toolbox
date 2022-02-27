import numpy as np
from typing import Any, List
import numpy.typing as npt


def simplify_lottery(
    values: npt.NDArray[np.int_], probs: npt.NDArray[np.float64]
):
    mask = probs > 0
    return values[mask], probs[mask]


def pad_zeros_1darray(arr: npt.NDArray[(Any,)], length: int):
    new_arr = np.zeros((length,))
    new_arr[: arr.shape[0]] = arr
    return new_arr


def stack_1darrays(arrs: List[npt.NDArray[(Any,)]]):
    """Stack 1d numpy arrays of different sizes into a 2d array, padded with zeros."""
    max_length = max([arr.shape[0] for arr in arrs])
    return np.array([pad_zeros_1darray(arr, max_length) for arr in arrs])


def decode_str_encoded_nparray(
    encoding: str, dtype: npt.DTypeLike = np.float64
):
    if encoding[0] != "[" or encoding[-1] != "]":
        raise ValueError(
            'An encoded numpy array must start with "[" and end with "]"'
        )
    return np.fromstring(encoding[1:-1], dtype=dtype, sep=" ")


def get_vals_from_encodings(vals: npt.ArrayLike):
    return stack_1darrays(
        [decode_str_encoded_nparray(val, dtype=int) for val in vals]
    )


def get_probs_from_encodings(probs: npt.ArrayLike):
    return stack_1darrays([decode_str_encoded_nparray(prob) for prob in probs])
