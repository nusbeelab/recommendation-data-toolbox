import itertools
import random
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from typing import Literal
import numpy.typing as npt

from recommendation_data_toolbox.rec.cf import CfRecommender


class DecisionTreeRecommender(CfRecommender):
    def __init__(
        self,
        rating_matrix: npt.NDArray[np.int_],
        subj_problem_ids: npt.NDArray[np.int_],
        subj_decisions: npt.NDArray[np.int_],
    ):
        super().__init__(rating_matrix, subj_problem_ids, subj_decisions)
        self.X = self.rating_matrix[:, self.subj_problem_ids]
        self.clf = DecisionTreeClassifier()

    def _rec(self, problem_id: int):
        y = self.rating_matrix[:, problem_id]
        self.clf.fit(self.X, y)
        return self.clf.predict([self.subj_decisions])[0]

    def rec(self, problem_ids: npt.NDArray[np.int_]):
        return np.array([self._rec(problem_id) for problem_id in problem_ids])


class NaiveBayesRecommender(CfRecommender):
    def __init__(
        self,
        rating_matrix: npt.NDArray[np.int_],
        subj_problem_ids: npt.NDArray[np.int_],
        subj_decisions: npt.NDArray[np.int_],
    ):
        super().__init__(rating_matrix, subj_problem_ids, subj_decisions)
        self.X = self.rating_matrix[:, self.subj_problem_ids]
        self.clf = GaussianNB()

    def _rec(self, problem_id: int):
        y = self.rating_matrix[:, problem_id]
        self.clf.fit(self.X, y)
        return self.clf.predict([self.subj_decisions])[0]

    def rec(self, problem_ids: npt.NDArray[np.int_]):
        return np.array([self._rec(problem_id) for problem_id in problem_ids])


def initialize_UV(m: int, n: int, k: int):
    U = np.random.rand(m, k)
    V = np.random.rand(n, k)
    return U, V


def is_converged(old: npt.NDArray, new: npt.NDArray, delta: float = 0.001):
    return np.sum(np.power(old - new, 2)) < delta


def mf_sgd(
    R: npt.NDArray,
    k: int = 5,
    max_steps: int = 5000,
    alpha: float = 0.005,
    beta: float = 0.01,
    delta: float = 0.0001,
):
    m, n = R.shape
    U, V = initialize_UV(m, n, k)
    S = [
        (i, j)
        for i, j in itertools.product(range(m), range(n))
        if not np.isnan(R[i, j])
    ]
    step_count = 0

    while step_count < max_steps:
        random.shuffle(S)
        U_old, V_old = U.copy(), V.copy()

        for i, j in S:
            e_ij = R[i, j] - np.dot(U_old[i, :], V_old[j, :])
            U[i, :] = U_old[i, :] + alpha * (
                e_ij * V_old[j, :] - beta * U_old[i, :]
            )
            V[j, :] = V_old[j, :] + alpha * (
                e_ij * U_old[i, :] - beta * V_old[j, :]
            )

        if is_converged(U, U_old, delta) and is_converged(V, V_old, delta):
            break

        step_count += 1

    return U, V


def least_squares_regression(A: npt.NDArray, y: npt.NDArray):
    mask = ~np.isnan(y)
    y = y[mask]
    A = A[mask]

    pinv = np.linalg.pinv(A)
    return np.dot(pinv, y)


def mf_als(
    R: npt.NDArray,
    k: int = 5,
    max_steps: int = 5000,
    delta: float = 0.000001,
):
    m, n = R.shape
    U, V = initialize_UV(m, n, k)
    step_count = 0
    while step_count < max_steps:
        U_old, V_old = U.copy(), V.copy()

        # U fixed
        for j in range(n):
            V[j, :] = least_squares_regression(U, R[:, j])

        # V fixed
        for i in range(m):
            U[i, :] = least_squares_regression(V, R[i, :])

        if is_converged(U, U_old, delta) and is_converged(V, V_old, delta):
            break

        step_count += 1

    return U, V


def get_mf_probs(
    U: npt.NDArray[np.int_],
    V: npt.NDArray[np.int_],
    subj_id: int,
    problem_ids: npt.NDArray[np.int_],
):
    return np.dot(V[problem_ids, :], U[subj_id, :])


class LatentFactorRecommender(CfRecommender):
    def __init__(
        self,
        rating_matrix: npt.NDArray[np.int_],
        subj_problem_ids: npt.NDArray[np.int_],
        subj_decisions: npt.NDArray[np.int_],
        optimization_method: Literal["sgd", "als"] = "als",
    ):
        super().__init__(rating_matrix, subj_problem_ids, subj_decisions)

        rating_vector = np.zeros((self.rating_matrix.shape[-1],))
        rating_vector[self.subj_problem_ids] = self.subj_decisions
        R = np.vstack((self.rating_matrix, rating_vector))

        self.U, self.V = (mf_sgd if optimization_method == "sgd" else mf_als)(R)

    def rec(self, problem_ids: npt.NDArray[np.int_]):
        return int(
            get_mf_probs(
                U=self.U, V=self.V, subj_id=-1, problem_ids=problem_ids
            )
            >= 0.5
        )
