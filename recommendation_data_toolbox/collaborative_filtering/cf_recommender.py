from typing import List
from recommendation_data_toolbox.lottery import DecisionHistory
from recommendation_data_toolbox.recommender import RecommenderWithHistory


class CfRecommender(RecommenderWithHistory):
    def __init__(
        self, history: DecisionHistory, records: List[DecisionHistory]
    ):
        super().__init__(history)
        self.records = list(records)
