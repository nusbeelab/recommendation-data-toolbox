from abc import abstractmethod
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from typing import List
import numpy.typing as npt

from recommendation_data_toolbox.lottery import Lottery, Problem, ProblemManager
from recommendation_data_toolbox.rec import Recommender


def get_lot_features(lot: Lottery):
    high = lot.objective_consequences.max()
    low = lot.objective_consequences.min()
    abs_diff = high - low
    most_likely = lot.objective_consequences[lot.probs.argmax()]
    least_likely = lot.objective_consequences[lot.probs.argmin()]
    consequence_num = len(lot.objective_consequences)
    expectation = sum(lot.objective_consequences * lot.probs)
    var = sum(lot.objective_consequences**2 * lot.probs) - expectation**2
    skewness = (
        (
            sum(lot.objective_consequences**3 * lot.probs)
            - 3 * expectation * var
            - expectation**3
        )
        / (var**1.5)
        if var != 0
        else 0
    )
    is_certain = len(lot.objective_consequences) == 1
    return np.array(
        [
            high,
            low,
            abs_diff,
            most_likely,
            least_likely,
            consequence_num,
            expectation,
            var,
            skewness,
            is_certain,
        ]
    )


def get_ordered_lottery(lot: Lottery):
    order = lot.objective_consequences.argsort()
    return Lottery(lot.objective_consequences[order], lot.probs[order])


def get_cum_probs(lot: Lottery):
    ordered_lot = get_ordered_lottery(lot)
    cum_probs = np.array(ordered_lot.probs)
    for i in range(1, len(cum_probs)):
        cum_probs[i] += cum_probs[i - 1]
    return ordered_lot.objective_consequences, cum_probs


def fosd(a: Lottery, b: Lottery):
    is_strictly_lt = False
    for x in set(
        list(a.objective_consequences) + list(b.objective_consequences)
    ):
        a_cum_prob = a.cum_prob(x)
        b_cum_prob = b.cum_prob(x)
        if a_cum_prob > b_cum_prob:
            return False
        if a_cum_prob < b_cum_prob:
            is_strictly_lt = True
    return is_strictly_lt


def get_features_per_problem(problem: Problem):
    a_features = get_lot_features(problem.a)
    b_features = get_lot_features(problem.b)
    consequence_num_ratio = b_features[5] / a_features[5]
    expectation_ratio = b_features[6] / a_features[6]
    fosd_a_over_b = fosd(problem.a, problem.b)
    fosd_b_over_a = fosd(problem.b, problem.a)
    return np.concatenate(
        [
            a_features,
            b_features,
            b_features - a_features,
            np.array(
                [
                    consequence_num_ratio,
                    expectation_ratio,
                    fosd_a_over_b,
                    fosd_b_over_a,
                ]
            ),
        ]
    )


def get_problem_features(probs: List[Problem]):
    return np.vstack([get_features_per_problem(prob) for prob in probs])


class ContentBasedRecommender(Recommender):
    def __init__(
        self,
        problem_manager: ProblemManager,
        subj_problem_ids: npt.NDArray[np.int_],
        subj_decisions: npt.NDArray[np.int_],
        clf,
    ):
        self.problem_manager = problem_manager
        self.clf = clf
        X = get_problem_features(
            self.problem_manager.convert_ids_to_problems(subj_problem_ids)
        )
        for x in X:
            if np.any(np.isnan(x)):
                print(x)
        y = subj_decisions
        self.clf.fit(X, y)

    def rec(self, problem_ids: npt.NDArray[np.int_]):
        input_X = get_problem_features(
            self.problem_manager.convert_ids_to_problems(problem_ids)
        )
        return self.clf.predict(input_X)

    @abstractmethod
    def get_feature_importance(self, **kwargs):
        pass


class ContentBasedDecisionTreeRecommender(ContentBasedRecommender):
    def __init__(
        self,
        problem_manager: ProblemManager,
        subj_problem_ids: npt.NDArray[np.int_],
        subj_decisions: npt.NDArray[np.int_],
    ):
        super().__init__(
            problem_manager,
            subj_problem_ids,
            subj_decisions,
            DecisionTreeClassifier(),
        )

    def get_feature_importance(self, **kwargs):
        return self.clf.feature_importances_


class ContentBasedRandomForestRecommender(ContentBasedRecommender):
    def __init__(
        self,
        problem_manager: ProblemManager,
        subj_problem_ids: npt.NDArray[np.int_],
        subj_decisions: npt.NDArray[np.int_],
    ):
        super().__init__(
            problem_manager,
            subj_problem_ids,
            subj_decisions,
            RandomForestClassifier(),
        )

    def get_feature_importance(self, **kwargs):
        return self.clf.feature_importances_


class ContentBasedGradientBoostingRecommender(ContentBasedRecommender):
    def __init__(
        self,
        problem_manager: ProblemManager,
        subj_problem_ids: npt.NDArray[np.int_],
        subj_decisions: npt.NDArray[np.int_],
    ):
        super().__init__(
            problem_manager,
            subj_problem_ids,
            subj_decisions,
            GradientBoostingClassifier(),
        )

    def get_feature_importance(self, **kwargs):
        return self.clf.feature_importances_


# class ContentBasedXgboostRecommender(ContentBasedRecommender):
#     def __init__(
#         self,
#         problem_manager: ProblemManager,
#         subj_problem_ids: npt.NDArray[np.int_],
#         subj_decisions: npt.NDArray[np.int_],
#     ):
#         super().__init__(
#             problem_manager,
#             subj_problem_ids,
#             subj_decisions,
#             XGBClassifier(),
#         )

#     def get_feature_importance(self, **kwargs):
#         return self.clf.feature_importances_


class ContentBasedNaiveBayesRecommender(ContentBasedRecommender):
    def __init__(
        self,
        problem_manager: ProblemManager,
        subj_problem_ids: npt.NDArray[np.int_],
        subj_decisions: npt.NDArray[np.int_],
    ):
        super().__init__(
            problem_manager, subj_problem_ids, subj_decisions, GaussianNB()
        )

    def get_feature_importance(self, test_problem_ids, test_decisions):
        X_test = get_problem_features(
            self.problem_manager.convert_ids_to_problems(test_problem_ids)
        )
        y_test = test_decisions
        r = permutation_importance(self.clf, X_test, y_test, random_state=1234)
        return r.importances_mean / sum(r.importances_mean)
