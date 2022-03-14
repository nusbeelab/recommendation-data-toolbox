import os
from typing import Iterable
import numpy as np
import numpy.typing as npt
from recommendation_cl_utils import CWD

from recommendation_data_toolbox.utils import stack_1darrays


def get_fullpath_to_datafile(filename):
    return os.path.join(CWD, "data", filename)


def decode_str_encoded_nparray(
    encoding: str, dtype: npt.DTypeLike = np.float64
):
    """Decodes a string encoding of a numpy array e.g. [1 2 3]."""
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


def snakecase_to_camelcase(snakecase: str):
    words = snakecase.split("_")
    words = [
        word.capitalize() if i > 0 else word for i, word in enumerate(words)
    ]
    return "".join(words)


def sort_lists_with_ordering(ordering: list, lists: Iterable[Iterable]):
    ordering_dict = {x: i for i, x in enumerate(ordering)}
    return zip(*sorted(zip(*lists), key=lambda x: ordering_dict[x[0]]))


def get_accuracy(actual: npt.NDArray, preds: npt.NDArray):
    if actual.shape != preds.shape:
        raise ValueError(f"{actual.shape} != {preds.shape}")
    return np.sum(actual == preds, axis=-1) / actual.shape[-1]
