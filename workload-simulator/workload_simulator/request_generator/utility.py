from abc import ABC, abstractmethod

from typing import List

from scipy.stats import beta

import warnings

from workload_simulator.request_generator.request import Request

class BaseUtilityAssigner(ABC):

    @abstractmethod
    def assign_utility(self, requests: List[dict]):
        pass

    @abstractmethod
    def short(self) -> str:
        pass




class ConstantUtilityShadowAssigner(BaseUtilityAssigner):

    """
    This utility assigner first uses the shadow assigner to determine a utility and then sets the profit to 1 (optimize for the number of requests)
    """

    def __init__(self, shadow_assigner: BaseUtilityAssigner):
        assert shadow_assigner is not None, "shadow assigner must not be None"
        self.shadow_assigner = shadow_assigner

    def assign_utility(self, requests: List[Request]):

        # first use the shadow assigner to assign the utility + config
        self.shadow_assigner.assign_utility(requests)

        for req in requests:
            req.request_info.utility_info["shadow_name"] = req.request_info.utility_info["name"]
            req.request_info.utility_info["name"] = __class__.__name__
            req.profit = 1

    def short(self):
        return f"equal-{self.shadow_assigner.short()}"



#class ConstantUtilityAssigner(BaseUtilityAssigner):
#
#    def assign_utility(self, requests: List[Request]):
#
#        cfg = self.config()
#
#        for req in requests:
#            req.set_utility(utility=1, info=cfg)
#
#    def short(self):
#        return "equal"
#
#    def config(self):
#        utility_config = {
#            "name": __class__.__name__,
#        }
#        return utility_config

class NormalizedCobbDouglasUtilityAssigner(BaseUtilityAssigner):


    def __init__(self, privacy_cost_elasticity, data_elasticity, scaling_beta_a, scaling_beta_b, use_balance_epsilon):
        self.privacy_cost_elasticity = privacy_cost_elasticity
        self.data_elasticity = data_elasticity

        self.scaling_beta_a = scaling_beta_a
        self.scaling_beta_b = scaling_beta_b

        self.normalization_factor = 1000

        # if true, then we take the max epsilon for each cost type (e.g., elephant, hare, mice) and use that for utility calculation) even though the mechanism may consume less budget
        self.balance_epsilon = use_balance_epsilon


    def short(self):
        return "ncd"

    def assign_utility(self, requests: List[Request]):

        scalings = beta.rvs(self.scaling_beta_a, self.scaling_beta_b, size=len(requests))

        def get_cost(req):
            return req.request_info.cost_original.adp["EpsDeltaDp"]
            #return req["request_info"]["cost"]["adp"]["EpsDeltaDp"]

        def get_cost_name(req):
            return req.request_info.cost_original.name

        def get_user_prob(req):
            return req.request_info.sampling_info.prob


        delta = get_cost(requests[0])["delta"]


        #utility_scaling = {m["mechanism"]["name"]: m["utility_weight"] for m in requests[0].workload_info["mechanism_mix"]}

        if self.balance_epsilon:
            cost_tmp = {}
            for m in requests[0].workload_info["mechanism_mix"]:
                for c in m["mechanism"]["cost_calibration"]["distribution"]:

                    if c["name"] not in cost_tmp:
                        cost_tmp[c["name"]] = set()
                    cost_tmp[c["name"]].add(c["epsilon"])

            for cost_name, epsilons in cost_tmp.items():
                if len(epsilons) > 1:
                    warnings.warn(f"cost {cost_name} has different epsilons: {epsilons}  -> we take the max epsilon for utility calculation")

            cost_lookup = {cost_name: max(epsilons) for cost_name, epsilons in cost_tmp.items()}



        #print(f"utility_scaling={utility_scaling}")


        utilities = []

        for req, scaling in zip(requests, scalings):
            cost = get_cost(req)
            assert cost["delta"] == delta, "delta should be the same for all requests to be able to use epsilon as proxy for utility"
            user_sampling_prob = get_user_prob(req)


            epsilon = cost_lookup[get_cost_name(req)] if self.balance_epsilon else cost["eps"]

            #privacy_cost_elasticity # beta
            #n_users_elasticity # alpha
            # scaling factor
            # cobb douglas production function
            Y = scaling * epsilon**self.privacy_cost_elasticity * user_sampling_prob **self.data_elasticity

            #cost_tmp = utility_scaling[req.request_info.mechanism["mechanism"]["name"]]

            utilities.append(max(1, int(Y * 1000)))


        s = sum(utilities)
        normalization_sum = self.normalization_factor * len(requests)
        utilities = [normalization_sum * float(u)/s for u in utilities]




        for req, utility in zip(requests, utilities):
            assert int(utility) >= 1, "utility should be at least 1 -> choose larger normalization"
            cfg = self.config()
            cfg ["normalization_sum"] = normalization_sum
            cfg["utility"] = utility # assign original utility

            req.set_utility(int(utility), cfg) # profit needs to be an integer

    def config(self):
        utility_config = {
            "name": __class__.__name__,
            "privacy_cost_elasticity": self.privacy_cost_elasticity,
            "data_elasticity": self.data_elasticity,
            "scaling": {
                "beta_a": self.scaling_beta_a,
                "beta_b": self.scaling_beta_b
            },
            "balance_epsilon": self.balance_epsilon
        }

        return utility_config
