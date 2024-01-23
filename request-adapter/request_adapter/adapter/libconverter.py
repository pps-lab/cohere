from datetime import timedelta
from request_adapter.adapter.tmlt.converter import convert_to_audodp as convert_to_audodp_tmlt
from request_adapter.adapter.opacus.converter import convert_to_autodp as convert_to_autodp_opacus


from workload_simulator.request_generator.mechanism import MechanismCost, MechanismInfo
from workload_simulator.request_generator.request import Request
from workload_simulator.request_generator.sampling import SamplingInfo

from tmlt.analytics.privacy_budget import PrivacyBudget, RhoZCDPBudget
from tmlt.analytics.query_builder import QueryBuilder, QueryExpr

class ConverterConfig:

    def __init__(self, active_time_window: timedelta, allocation_interval: timedelta, alphas=None):
        self.request_id = 0
        self.K = active_time_window // allocation_interval

        if alphas is None:
            alphas = [1.5, 1.75, 2, 2.5, 3, 4, 5, 6, 8, 16, 32, 64, 1e6] # default
        self.ALPHAS = alphas

    def new_request_id(self):
        # NOTE: would have to obtained from central request id generator
        self.request_id += 1
        return self.request_id

    def current_round_id(self):
        # NOTE: would need to be connected to the current round id
        return 1


def create_tumult_request(session, query_expr: QueryExpr, budget: RhoZCDPBudget, converter_config: ConverterConfig, population_dnf=None, population_prob: float = 1.0, utility: int = 1):

    mech_autodp, mech_autodp_amplified, mechanisms_info = convert_to_audodp_tmlt(session=session, query_expr=query_expr, budget=budget, population_prob=population_prob)
    return _create_request(converter_config=converter_config, mech_autodp=mech_autodp, mech_autodp_amplified=mech_autodp_amplified, population_dnf=population_dnf, population_prob=population_prob, mechanisms_info=mechanisms_info, utility=utility)

def create_opacus_request(optimizer, n_batches_per_epoch: int, epochs: int, converter_config: ConverterConfig, population_dnf=None, population_prob: float = 1.0, utility: int = 1):

    mech_autodp, mech_autodp_amplified, mechanisms_info = convert_to_autodp_opacus(optimizer=optimizer, n_batches=n_batches_per_epoch, epochs=epochs, population_prob=population_prob)
    return _create_request(converter_config=converter_config, mech_autodp=mech_autodp, mech_autodp_amplified=mech_autodp_amplified, population_dnf=population_dnf, population_prob=population_prob, mechanisms_info=mechanisms_info, utility=utility)


def _create_request(converter_config: ConverterConfig, mech_autodp, mech_autodp_amplified, population_dnf, population_prob, mechanisms_info, utility):

    cost_name = "from-adapter"
    mcost = MechanismCost(name=cost_name)
    mcost.set_cost_rdp(mechanism=mech_autodp, alphas=converter_config.ALPHAS)

    mcost_amplified = MechanismCost(name=cost_name)
    mcost_amplified.set_cost_rdp(mechanism=mech_autodp_amplified, alphas=converter_config.ALPHAS)

    sampling_info = SamplingInfo(name="manual", prob=population_prob)

    mechanism_info = MechanismInfo(
                        mechanism=mechanisms_info,
                        cost_poisson_amplified=mcost_amplified,
                        cost_original=mcost,
                        sampling_info=sampling_info)


    if population_dnf is None:
        population_dnf = {} # TODO: could improve the interface here

    r = Request(
            request_id=converter_config.new_request_id(),
            created=converter_config.current_round_id(),
            dnf=population_dnf,
            request_info=mechanism_info,
            workload_info=None)

    r.set_utility(utility=utility, info={"name": "manual"})
    r.set_n_users(converter_config.K) # called n_users for backward compatibility, but actually it is the number of rounds in the active window
    r.set_request_cost(r.request_info.cost_poisson_amplified.rdp)

    return r