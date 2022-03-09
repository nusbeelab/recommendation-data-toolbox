import numpy as np


def roll_fill_last_dim(arr: np.ndarray, shift, fill_value=0.0):
    result = np.empty_like(arr)
    if shift > 0:
        result[..., :shift] = fill_value
        result[..., shift:] = arr[..., :-shift]
    elif shift < 0:
        result[..., shift:] = fill_value
        result[..., :shift] = arr[..., -shift:]
    else:
        result[...] = arr
    return result
