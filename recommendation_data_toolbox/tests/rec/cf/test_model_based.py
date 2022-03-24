import unittest

import numpy as np

from recommendation_data_toolbox.rec.cf.model_based import (
    initialize_UV,
    mf_sgd,
    mf_als,
)


class TestLatentFactor(unittest.TestCase):
    def test_initializeUV(self):
        U, V = initialize_UV(m=4, n=3, k=2)
        self.assertEqual(U.shape, (4, 2))
        self.assertEqual(V.shape, (3, 2))
        self.assertTrue(np.all(np.logical_and(U >= 0, U <= 1)))
        self.assertTrue(np.all(np.logical_and(V >= 0, V <= 1)))

    def test_mfSgd(self):
        R = np.array(
            [
                [1, 0, 1, 0],
                [1, 0, 1, 0],
                [0, 0, 0, 1],
                [1, 0, 1, 1],
                [0, np.nan, np.nan, 1],
            ]
        )
        U, V = mf_sgd(R, k=3, delta=0.00000001)
        R_approx = np.dot(U, V.T)

        mask = ~(np.isnan(R) | np.isnan(R_approx))
        self.assertTrue(np.allclose(R[mask], R_approx[mask], atol=0.1))

    def test_mfWals(self):
        R = np.array(
            [
                [1, 0, 1, 0],
                [1, 0, 1, 0],
                [0, 0, 0, 1],
                [1, 0, 1, 1],
                [0, np.nan, np.nan, 1],
            ]
        )
        U, V = mf_als(R, k=3, delta=0.00000001)
        R_approx = np.dot(U, V.T)

        mask = ~(np.isnan(R) | np.isnan(R_approx))
        self.assertTrue(np.allclose(R[mask], R_approx[mask], atol=0.1))


if __name__ == "__main__":
    unittest.main()
