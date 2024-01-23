from workload_simulator.request_generator.partitioning import BasePopulationSelection

from workload_simulator.request_generator.mechanism import MechanismInfo, MechanismWrapper

from dataclasses import dataclass

import random

from typing import List

from workload_simulator.schema.schema import Schema


@dataclass
class Request:
    request_id: int
    created: int # round id
    dnf: dict
    request_info: MechanismInfo
    workload_info: dict

    # supplied later
    profit: int = None
    n_users: int = None
    request_cost: dict = None

    def set_utility(self, utility, info):
        assert isinstance(utility, int), f"utility must be an integer, but is {type(utility)}"
        self.profit = utility
        self.request_info.utility_info = info


    def set_n_users(self, n_users):
        assert isinstance(n_users, int), f"n_users must be an integer, but is {type(n_users)}"
        self.n_users = n_users

    def set_request_cost(self, request_cost):
        assert isinstance(request_cost, dict), f"request_cost must be a dict, but is {type(request_cost)}"
        self.request_cost = request_cost



@dataclass
class Category:

    population_gen: BasePopulationSelection
    mechanism_gen: MechanismWrapper
    weight: int

    def info(self):
        return {
            "population": self.population_gen.info(),
            "mechanism": self.mechanism_gen.info(),
            "weight": self.weight
        }

    def generate_request(self, request_id: int, round_id: int, workload_info: dict):

        population_dnf, population_domain_size = self.population_gen.generate()

        alphas = workload_info["cost_config"]["alphas"]

        mechanism_info = self.mechanism_gen.generate(alphas)
        mechanism_info.selection = {
            "n_conjunctions": len(population_dnf["conjunctions"]),
            "n_virtual_blocks": population_domain_size
        }

        r = Request(
            request_id=request_id,
            created=round_id,
            dnf=population_dnf,
            request_info=mechanism_info,
            workload_info=workload_info
        )

        return r

@dataclass
class Workload:

    name: str
    schema: Schema
    distribution: List[Category]


    _request_id_ctr: int = 0

    def __post_init__(self):

        self.weights = [cat.weight for cat in self.distribution]
        self.population = list(range(len(self.weights)))

        self.workload_info = {
            "name": self.name,
            "size": None,
            "mechanism_mix": [d.info() for d in self.distribution],
            "cost_config": self.schema.cost_config()
        }

    def empty_cost(self):
        rdp = len(self.workload_info["cost_config"]["alphas"]) * [0.0]
        return {
            "Rdp": {
                "eps_values": rdp
            }
        }

    def generate_request(self, round_id: int):

        request_category_idx = random.choices(population=self.population, weights=self.weights, k=1)[0]

        request = self.distribution[request_category_idx].generate_request(request_id=self._request_id_ctr, round_id=round_id, workload_info=self.workload_info)

        self._request_id_ctr += 1

        return request
