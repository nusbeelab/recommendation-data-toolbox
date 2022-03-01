import unittest
import numpy as np

from param_estimation.utils import decode_str_encoded_nparray


class TestUtils(unittest.TestCase):
    def test_decodeNparray(self):
        self.assertTrue(
            np.array_equal(
                decode_str_encoded_nparray("[1 2]", dtype=int), np.array([1, 2])
            )
        )
