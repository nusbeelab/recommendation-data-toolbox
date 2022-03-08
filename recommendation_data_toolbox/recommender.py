from abc import ABCMeta, abstractmethod
from random import random
from typing import Optional

from recommendation_data_toolbox.lottery import DecisionHistory, LotteryPair


class Recommender(metaclass=ABCMeta):
    @abstractmethod
    def rec(self, lottery_pair: LotteryPair) -> Optional[bool]:
        pass


class NoneRecommender(Recommender):
    def rec(self, lottery_pair: LotteryPair):
        return None


class RandomRecommender(Recommender):
    def rec(self, lottery_pair: LotteryPair):
        return random.random() < 0.5


class RecommenderWithHistory(Recommender):
    def __init__(self, history: DecisionHistory):
        self.history = history
