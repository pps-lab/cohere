
import math
from typing import Dict, List, Union

from autodp import transformer_zoo
from tmlt.analytics._noise_info import _noise_from_measurement, _NoiseMechanism
from tmlt.analytics.privacy_budget import RhoZCDPBudget
from tmlt.analytics.query_builder import QueryExpr
from tmlt.core.measures import RhoZCDP
from tmlt.core.utils.exact_number import ExactNumber
from tmlt.core.utils.parameters import calculate_noise_scale
from workload_simulator.request_generator.mechanism import (
    DiscreteGaussianMechanism, ExponentialMechanism, GaussianMechanism)


def _measurement_to_autodp(budget: RhoZCDPBudget, info_tmlt: Dict, population_prob: int = 1, sensitivity_expected: float = None):

    assert isinstance(budget, RhoZCDPBudget)

    mtype = info_tmlt["noise_mechanism"]

    if mtype == _NoiseMechanism.DISCRETE_GAUSSIAN or mtype == _NoiseMechanism.GAUSSIAN:

        if sensitivity_expected is not None:
            sigma_expected = calculate_noise_scale(d_in=ExactNumber.from_float(sensitivity_expected, round_up=True), d_out=budget.value, output_measure=RhoZCDP())
            sigma_squared_expected = sigma_expected ** "2"
            sigma_squared_expected = sigma_squared_expected.to_float(round_up=True)
            sigma_squared_tmlt = info_tmlt["noise_parameter"]

            assert math.isclose(sigma_squared_expected, sigma_squared_tmlt), f"Error gaussian mechanism translation: expected sigma_squared does not match derived sigma_squared  expected={sigma_squared_expected}  derived={sigma_squared_tmlt}"

        sigma = calculate_noise_scale(d_in=ExactNumber("1"), d_out=budget.value, output_measure=RhoZCDP())

        if mtype == _NoiseMechanism.DISCRETE_GAUSSIAN:
            wrapper = DiscreteGaussianMechanism()
            mech_autodp, info = wrapper.create(sigma=sigma.to_float(round_up=True))
            mech_autodp_amplified = wrapper.get_poisson_amplified_mechanism(mech_autodp, amplify_poisson_lambda=population_prob)
            return mech_autodp, mech_autodp_amplified, info
        elif mtype == _NoiseMechanism.GAUSSIAN:
            wrapper = GaussianMechanism()
            mech_autodp, info = wrapper.create(sigma=sigma.to_float(round_up=True))
            mech_autodp_amplified = wrapper.get_poisson_amplified_mechanism(mech_autodp, amplify_poisson_lambda=population_prob)
            return mech_autodp, mech_autodp_amplified, info
        else:
            raise ValueError("unreachable")


    elif mtype == _NoiseMechanism.EXPONENTIAL:
        eps_tmlt = info_tmlt["noise_parameter"]

        #rho_tmlt = budget.value.to_float(round_up=False)

        # eps_tmlt = sqrt(8 * rho_tmlt) / sensitivity_tmlt
        print(f"eps_tmlt: {eps_tmlt}")
        print(f"budget.value: {budget.value}")

        if sensitivity_expected is not None:
            sensitivity_tmlt = (8 * budget.value) ** "1/2" / ExactNumber.from_float(eps_tmlt, round_up=False)
            sensitivity_actual = sensitivity_tmlt.to_float(round_up=True)
            assert math.isclose(sensitivity_expected, sensitivity_actual), f"Error exponential mechanism translation: expected sensitivity does not match derived sensitivity  expected={sensitivity_expected}  derived={sensitivity_actual}"

        # for sensitivity 1, this is the auto dp noise parameter
        eps_autodp = (8 * budget.value) ** "1/2"
        #print(f"--> AutoDP: ExponentialMechanism(eps={eps_autodp})")

        wrapper = ExponentialMechanism()

        mech_autodp, info = wrapper.create(epsilon=eps_autodp.to_float(round_up=True))
        mech_autodp_amplified = wrapper.get_poisson_amplified_mechanism(mech_autodp, amplify_poisson_lambda=population_prob)
        return mech_autodp, mech_autodp_amplified, info

        # eps_autodp = eps_tmlt * sensitivity_tmlt
        # sensitivity_tmlt = sqrt(8 * rho_tmlt) / eps_tmlt
        # => eps_autodp = sqrt(8 * rho_tmlt)

        # For Rho zCDP: (in tmlt/core/measurements/aggregations.py: 1838)
        #            d_out: privacy budget (rho) given @ evaluate
        #            d_mid: n_records per id (sensitivity_tmlt)
        #         epsilon = (8 * d_out) ** "1/2" / d_mid

    else:
        raise ValueError(f"unknown mechanism: {mtype}")


def convert_to_audodp(session, query_expr: QueryExpr, budget: RhoZCDPBudget, population_prob: int = 1, expected_sensitivity: Union[float, List[float]] = None):

    assert isinstance(budget, RhoZCDPBudget)

    # pylint: disable=W0212
    measurement, adjusted_budget = session._compile_and_get_budget(query_expr, budget)
    mlist = _noise_from_measurement(measurement)

    if expected_sensitivity is None:
        expected_sensitivity = [None] * len(mlist)
    elif not isinstance(expected_sensitivity, list):
        expected_sensitivity = [expected_sensitivity]

    assert len(mlist) == len(expected_sensitivity), "expected sensitivity must be a list of the same length as the number of measurements"

    # NOTE: In `tmlt/core/measurements/aggregations.py`, the aggregations resulting in multiple measurements
    #       (e.g., average: sum + count, variance: sum  sum of squares + count)
    #       are realized by splitting budget equally between the measurements.

    budget_per_measurement = RhoZCDPBudget(adjusted_budget.value / len(mlist))

    mechanisms_autodp = []
    mechanisms_autodp_amplified = []
    mechanisms_info = []

    for info, s in zip(mlist, expected_sensitivity, strict=True):
        print(f"  info: {info}  sensitivity_expected: {s}")
        mech_autodp, mech_autodp_amplified, minfo = _measurement_to_autodp(budget=budget_per_measurement, info_tmlt=info, population_prob=population_prob, sensitivity_expected=s)
        mechanisms_autodp.append(mech_autodp)
        mechanisms_autodp_amplified.append(mech_autodp_amplified)
        mechanisms_info.append(minfo)

    compose = transformer_zoo.Composition()
    mech_autodp_composed = compose(mechanisms_autodp, len(mechanisms_autodp) * [1])
    mech_autodp_amplified_composed = compose(mechanisms_autodp_amplified, len(mechanisms_autodp_amplified) * [1])

    return mech_autodp_composed, mech_autodp_amplified_composed, mechanisms_info