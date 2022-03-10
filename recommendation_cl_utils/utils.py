import os
import numpy as np
import numpy.typing as npt
from recommendation_cl_utils import CWD

from recommendation_data_toolbox.utils import stack_1darrays


def get_full_data_filepath(filename):
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


def get_accuracy(actual: npt.ArrayLike, preds: npt.ArrayLike):
    if len(actual) != len(preds):
        raise ValueError(
            "Two array-like objects must be of the same size to compute accuracy."
        )
    actual = np.array(actual)
    preds = np.array(preds)
    return sum(actual == preds) / len(actual)
