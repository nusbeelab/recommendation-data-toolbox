import unittest
import numpy as np

from recommendation_data_toolbox.utils import (
    decode_str_encoded_nparray,
    stack_1darrays,
)


class TestUtils(unittest.TestCase):
    def test_stack1darrays(self):
        arrs = [np.array([1]), np.array([1, 2, 3]), np.array([1, 2])]
        expected = np.array([[1, 0, 0], [1, 2, 3], [1, 2, 0]])
        self.assertTrue(np.array_equal(stack_1darrays(arrs), expected))

    def test_decodeNparray(self):
        self.assertTrue(
            np.array_equal(
                decode_str_encoded_nparray("[1 2]", dtype=int), np.array([1, 2])
            )
        )
