import unittest
import numpy as np

from recommendation_data_toolbox.models.prob_weight import (
    identity_pwf,
    tk1992_cpt_pwf,
    tk1992_pt_pwf,
)


class TestProbWeight(unittest.TestCase):
    def test_identityPwf(self):
        self.assertTrue(
            np.allclose(
                identity_pwf((), np.array([[0.2, 0.3, 0.5], [0.1, 0.6, 0.3]])),
                np.array([[0.2, 0.3, 0.5], [0.1, 0.6, 0.3]]),
            )
        )

    def test_tk1992PtPwf(self):
        actual = tk1992_pt_pwf(
            (0.5,), np.array([[0.2, 0.3, 0.5], [0.1, 0.6, 0.3]])
        )
        expected = np.array(
            [
                [0.2484519975, 0.2857908849, 0.35355339059],
                [0.19764235376, 0.39125077002, 0.2857908849],
            ]
        )
        self.assertTrue(np.allclose(actual, expected))

    def test_tk1992CptPwf(self):
        actual = tk1992_cpt_pwf(
            (0.5,), np.array([[0.2, 0.3, 0.5], [0.1, 0.6, 0.3]])
        )
        expected = np.array(
            [
                [0.503096005, 0.14335060441, 0.35355339059],
                [0.40707293871, 0.30713617638, 0.2857908849],
            ]
        )
        self.assertTrue(np.allclose(actual, expected))
