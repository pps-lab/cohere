from abc import ABC, abstractmethod
import math
import random

class Defaults:

    # NOTE: These are the old values used in Cohere V1
    #@staticmethod
    #def mice50_hare25_elephant25():
    #    """
    #    Three types of requests with varying epsilon and a fixed delta (1.0e-9):
    #    - 50% mice with epsilon 0.005
    #    - 25% hare with epsilon 0.2
    #    - 25% elephant with epsilon 0.8
    #    """
    #    return CategoricalCostCalibration([
    #        {"name": "mice", "prob": 0.5, "epsilon": 0.005, "delta": 1.0e-9},
    #        {"name": "hare", "prob": 0.25, "epsilon": 0.2, "delta": 1.0e-9},
    #        {"name": "elephant", "prob": 0.25, "epsilon": 0.8, "delta": 1.0e-9},
    #    ])

    @staticmethod
    def mice_hare_elephant_v2(mice=1/3, hare=1/3, elephant=1/3):
        """
        Three types of requests with varying epsilon and a fixed delta (1.0e-9):
        - 1/3 mice with epsilon 0.05
        - 1/3 hare with epsilon 0.2
        - 1/3 elephant with epsilon 0.75
        """
        prob_sum = mice + hare + elephant
        assert prob_sum == 1.0 or math.isclose(prob_sum, 1.0), "probabilities do not sum to 1.0"


        return CategoricalCostCalibration([
            {"name": "mice", "prob": mice, "epsilon": 0.05, "delta": 1.0e-9},
            {"name": "hare", "prob": hare, "epsilon": 0.2, "delta": 1.0e-9},
            {"name": "elephant", "prob": elephant, "epsilon": 0.75, "delta": 1.0e-9},
        ])

    @staticmethod
    def mice_hare_elephant_cheap(mice=1/3, hare=1/3, elephant=1/3):
        """
        Three types of requests with varying epsilon and a fixed delta (1.0e-9):
        - 1/3 mice with epsilon 0.01
        - 1/3 hare with epsilon 0.1
        - 1/3 elephant with epsilon 0.25
        """
        prob_sum = mice + hare + elephant
        assert prob_sum == 1.0 or math.isclose(prob_sum, 1.0), "probabilities do not sum to 1.0"


        return CategoricalCostCalibration([
            {"name": "mice", "prob": mice, "epsilon": 0.01, "delta": 1.0e-9},
            {"name": "hare", "prob": hare, "epsilon": 0.1, "delta": 1.0e-9},
            {"name": "elephant", "prob": elephant, "epsilon": 0.25, "delta": 1.0e-9},
        ])


class BaseCostCalibration(ABC):

    @abstractmethod
    def sample(self):
        pass

    @abstractmethod
    def info(self) -> dict:
        pass

class CategoricalCostCalibration(BaseCostCalibration):

    # the mechanism wrapper is just what I already had in: workload/mechanisms.py
    def __init__(self, cost_distribution):

        # cost_distribution =   [{"name": "mice", "prob": 0.5, "epsilon": 0.005, "delta": 1.0e-9}, {"name": "hare", "prob": 0.25, "epsilon": 0.2, "delta": 1.0e-9}, {"name": "elephant", "prob": 0.25", epsilon": 0.8, "delta": 1.0e-9}]

        self.cost_distribution = cost_distribution

        self.weights = [b["prob"] for b in cost_distribution]
        assert sum(self.weights) == 1.0 or math.isclose(sum(self.weights), 1.0), "probabilities do not sum to 1.0"

        self.population = list(range(len(self.weights)))


    def sample(self):
        idx = random.choices(population=self.population, weights=self.weights, k=1)[0]
        x = self.cost_distribution[idx]

        return x["epsilon"], x["delta"], x["name"]


    def info(self) -> dict:
        return {
            "name": __class__.__name__,
            "distribution": self.cost_distribution
        }
