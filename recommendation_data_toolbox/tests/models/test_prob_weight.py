import unittest
import numpy as np

from recommendation_data_toolbox.models.prob_weight import (
    identity_pwf,
    tversky_kahneman_1992_pwf,
)


class TestProbWeight(unittest.TestCase):
    def test_identityPwf(self):
        self.assertTrue(
            np.allclose(
                identity_pwf((), np.array([[0.2, 0.3, 0.5], [0.1, 0.6, 0.3]])),
                np.array([[0.2, 0.3, 0.5], [0.1, 0.6, 0.3]]),
            )
        )

    def test_tverskyKahneman1992Pwf(self):
        actual = tversky_kahneman_1992_pwf(
            (0.5,), np.array([[0.2, 0.3, 0.5], [0.1, 0.6, 0.3]])
        )
        expected = np.array(
            [
                [0.2484519975, 0.2857908849, 0.35355339059],
                [0.19764235376, 0.39125077002, 0.2857908849],
            ]
        )
        self.assertTrue(np.allclose(actual, expected))
