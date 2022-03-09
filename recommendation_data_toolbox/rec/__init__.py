from abc import ABCMeta, abstractmethod
from random import random
from typing import Optional

from recommendation_data_toolbox.lottery import LotteryPair


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
