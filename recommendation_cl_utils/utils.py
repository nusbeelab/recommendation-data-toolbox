import numpy as np
import numpy.typing as npt

from recommendation_data_toolbox.utils import stack_1darrays


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
