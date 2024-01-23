from abc import ABC, abstractmethod
import random

from dataclasses import dataclass

# Default Options

class Defaults:

    @staticmethod
    def single(p):

        assert p > 0 and p <= 1, f"p must be between 0 and 1 but is: {p}"

        return CategoricalSampling([
            {"name": f"{int(p * 100)}%", "prob": 1, "fraction": p},
        ])


    #@staticmethod
    #def equal_p10_p100():
    #    """
    #    With equal (50-50) probability, request 10% or 100% of the users.
    #    """
    #    return CategoricalSampling([
    #        {"name": "10%", "prob": 0.5, "fraction": 0.1},
    #        {"name": "100%", "prob": 0.5, "fraction": 1},
    #    ])
#
    #@staticmethod
    #def equal_p20_p100():
    #    """
    #    With equal (50-50) probability, request 20% or 100% of the users.
    #    """
    #    return CategoricalSampling([
    #        {"name": "20%", "prob": 0.5, "fraction": 0.2},
    #        {"name": "100%", "prob": 0.5, "fraction": 1},
    #    ])

    @staticmethod
    def equal_p25_p100():
        """
        With equal (50-50) probability, request 25% or 100% of the users.
        """
        return CategoricalSampling([
            {"name": "25%", "prob": 0.5, "fraction": 0.25},
            {"name": "100%", "prob": 0.5, "fraction": 1},
        ])

    #@staticmethod
    #def equal_p30_p100():
    #    """
    #    With equal (50-50) probability, request 30% or 100% of the users.
    #    """
    #    return CategoricalSampling([
    #        {"name": "30%", "prob": 0.5, "fraction": 0.3},
    #        {"name": "100%", "prob": 0.5, "fraction": 1},
    #    ])
#
    #@staticmethod
    #def equal_p40_p100():
    #    """
    #    With equal (50-50) probability, request 40% or 100% of the users.
    #    """
    #    return CategoricalSampling([
    #        {"name": "40%", "prob": 0.5, "fraction": 0.4},
    #        {"name": "100%", "prob": 0.5, "fraction": 1},
    #    ])

@dataclass
class SamplingInfo:
    name: str
    prob: float

class BaseSampling(ABC):

    @abstractmethod
    def sample(self) -> SamplingInfo:
        pass

    @abstractmethod
    def info(self):
        pass

class CategoricalSampling(BaseSampling):

    def __init__(self, rate_distribution):
        #expects a categorical distribution: rate_distribution = [{"name": "10%", "prob": 0.3, "fraction": 0.1}, {"name": "100%", "prob": 0.7, "fraction": 1}]
        self.rate_distribution = rate_distribution
        self.weights = [b["prob"] for b in self.rate_distribution]
        self.population = list(range(len(self.weights)))

    def _sample_categorical_fraction(self):
        idx = random.choices(population=self.population, weights=self.weights, k=1)[0]
        fraction = self.rate_distribution[idx]["fraction"]
        return fraction, self.rate_distribution[idx]["name"]

    def sample(self):
        fraction, name = self._sample_categorical_fraction()
        return SamplingInfo(name=name, prob=fraction)

    def info(self):
        return {
            "name": self.__class__.__name__,
            "distribution": self.rate_distribution
        }
