

from typing import Callable, Dict, List, Tuple

import pytest
import math
import importlib
from pyspark.sql import DataFrame
from tmlt.analytics.constraints import (MaxGroupsPerID, MaxRowsPerGroupPerID,
                                        MaxRowsPerID)
from tmlt.analytics.privacy_budget import (ApproxDPBudget, PureDPBudget,
                                           RhoZCDPBudget)
from tmlt.analytics.query_builder import QueryBuilder, QueryExpr
from request_adapter.adapter.tmlt.applications.base import BaseApplication, BaseScenario, InsufficientBudgetException
from request_adapter.adapter.tmlt.applications.netflix import AverageQuery, CountQuery, NetflixScenario, VarianceStdDevQuery


from autodp import transformer_zoo
from autodp import rdp_bank
from request_adapter.adapter.tmlt.applications.tmlt_tutorial import TumultTutorialScenario

@pytest.fixture(scope="module")
def scenario():
    print("\nSetup before the module")
    yield NetflixScenario()

@pytest.fixture(scope="module")
def scenario_tutorial_tmlt():
    print("\nSetup before the module")
    yield TumultTutorialScenario()

@pytest.fixture
def expo_based_applications():

    class FloatExpoQuery(BaseApplication):

        def queries(self) -> List[Callable[[Dict[str, DataFrame]], Tuple[str, QueryExpr, float]]]:

            def min_float_s1(session) -> Tuple[str, QueryExpr]:
                query = (
                    QueryBuilder("metrics")
                        .enforce(MaxRowsPerID(1)) # => sensitivity 1
                        .min("metric1", low=-10, high=100)
                )
                sensitivity = 1
                return "min_float_s1", query, sensitivity

            def min_float_s3(session) -> Tuple[str, QueryExpr]:
                query = (
                    QueryBuilder("metrics")
                        .enforce(MaxRowsPerID(3)) # => sensitivity 1
                        .min("metric1", low=-10, high=100)
                )
                sensitivity = 3
                return "min_float_s3", query, sensitivity

            return [min_float_s1, min_float_s3]

    class IntExpoQuery(BaseApplication):

        def queries(self) -> List[Callable[[Dict[str, DataFrame]], Tuple[str, QueryExpr, float]]]:
            def max_int_s1(session) -> Tuple[str, QueryExpr]:
                query = (
                    QueryBuilder("sessions")
                        .enforce(MaxRowsPerID(1)) # => sensitivity 1
                        .max("playbackLengthMin", low=0, high=600)
                )
                sensitivity = 1
                return "max_int_s1", query, sensitivity

            def max_int_s5(session) -> Tuple[str, QueryExpr]:
                query = (
                    QueryBuilder("sessions")
                        .enforce(MaxRowsPerID(5)) # => sensitivity 5
                        .max("playbackLengthMin", low=0, high=1000)
                )
                sensitivity = 5
                return "max_int_s5", query, sensitivity

            return [max_int_s1, max_int_s5]

    return [FloatExpoQuery, IntExpoQuery]

def test_tmlt_privacy_filter_ok(scenario, expo_based_applications):

    rho = 0.2
    per_query_budget = RhoZCDPBudget(rho)

    applications = [app(scenario) for app in expo_based_applications]
    n_queries_total = sum(len(app.queries()) for app in applications)

    scenario.create_session(budget=RhoZCDPBudget(n_queries_total * rho))

    for app in applications:
        n_queries = len(app.queries())
        app.execute(scenario.session, n_queries * [per_query_budget], skip_exec=False)


def test_tmlt_privacy_filter_err(scenario, expo_based_applications):

    rho = 0.2
    per_query_budget = RhoZCDPBudget(rho)

    applications = [app(scenario) for app in expo_based_applications]
    n_queries_total = sum(len(app.queries()) for app in applications)

    scenario.create_session(budget=RhoZCDPBudget(n_queries_total * rho - 0.05))

    for app in applications[:-1]:
        n_queries = len(app.queries())
        app.execute(scenario.session, n_queries * [per_query_budget], skip_exec=False)

    with pytest.raises(InsufficientBudgetException) as _exc_info:
        app = applications[-1]
        n_queries = len(app.queries())
        app.execute(scenario.session, n_queries * [per_query_budget], skip_exec=False)


def test_convert_expo_to_autodp(scenario, expo_based_applications):

    applications = [app(scenario) for app in expo_based_applications]

    scenario.create_session(budget=RhoZCDPBudget(float("inf")))

    for rho in [0.1, 0.5, 1, 3]:
        per_query_budget = RhoZCDPBudget(rho)

        for app in applications:
            n_queries = len(app.queries())
            app.execute(scenario.session, n_queries * [per_query_budget], skip_exec=True)


def test_convert_gaussian_to_autodp(scenario):

    applications = [app(scenario) for app in [AverageQuery, CountQuery, VarianceStdDevQuery]]

    scenario.create_session(budget=RhoZCDPBudget(float("inf")))

    for rho in [0.1, 0.5, 1, 3]:
        per_query_budget = RhoZCDPBudget(rho)

        for app in applications:
            n_queries = len(app.queries())
            app.execute(scenario.session, n_queries * [per_query_budget], skip_exec=True)

def test_convert_all_netflix(scenario):

    module = importlib.import_module("request_adapter.adapter.tmlt.applications.netflix")
    applications = []

    scenario.create_session(budget=RhoZCDPBudget(float("inf")))

    for name, obj in module.__dict__.items():
        if isinstance(obj, type) and issubclass(obj, BaseApplication) and obj != BaseApplication:
            applications.append(obj(scenario))


    for rho in [0.1, 0.5, 1, 3]:
        per_query_budget = RhoZCDPBudget(rho)

        for app in applications:
            n_queries = len(app.queries())
            app.execute(scenario.session, n_queries * [per_query_budget], skip_exec=True, enforce_sensitivity_check=True)


def test_convert_all_tmlt_tutorial(scenario_tutorial_tmlt):


    module = importlib.import_module("request_adapter.adapter.tmlt.applications.tmlt_tutorial")
    applications = []

    scenario_tutorial_tmlt.create_session(budget=RhoZCDPBudget(float("inf")))

    for _name, obj in module.__dict__.items():
        if isinstance(obj, type) and issubclass(obj, BaseApplication) and obj != BaseApplication:
            applications.append(obj(scenario_tutorial_tmlt))


    for rho in [0.1, 0.5, 1, 3]:
        per_query_budget = RhoZCDPBudget(rho)

        for app in applications:
            n_queries = len(app.queries())
            app.execute(scenario_tutorial_tmlt.session, n_queries * [per_query_budget], skip_exec=False, enforce_sensitivity_check=True)



def test_e2e_conversion_single(scenario):

    class Q1Expo(BaseApplication):
        def queries(self):
            def max_int(_session):
                query = (
                    QueryBuilder("sessions")
                        .enforce(MaxRowsPerID(1)) # => sensitivity 5
                        .max("playbackLengthMin", low=0, high=600)
                )
                sensitivity = 1
                return "max_int", query, sensitivity

            return [max_int]

    total_rho = 0.25
    total_budget_tmlt = RhoZCDPBudget(total_rho)

    alphas = [1.5, 1.75, 2, 2.5, 3, 4, 5, 6, 8, 16, 32, 64, 1e6]

    total_budget_autodp = [rdp_bank.RDP_zCDP(params={"rho": total_rho}, alpha=alpha) for alpha in alphas]

    scenario.create_session(budget=total_budget_tmlt)

    app = Q1Expo(scenario)
    h = app.execute(scenario.session, [total_budget_tmlt]) # use complete budget


    mech_autodp_composed, mech_autodp_amplified_composed, mechanisms_info = next(iter(h.values()))

    n_times_possible = max(b/mech_autodp_composed.get_RDP(alpha) for alpha, b in zip(alphas, total_budget_autodp, strict=True))

    costs = [mech_autodp_composed.get_RDP(alpha) for alpha in alphas]
    print(f"COSTS={costs}")
    print(f"BUDGET={total_budget_autodp}")

    assert math.isclose(n_times_possible, 1), f"n_times_possible={n_times_possible}  (expected to be close to 1)"



def test_e2e_conversion(scenario):

    class Q1Expo(BaseApplication):
        def queries(self):
            def max_int(_session):
                query = (
                    QueryBuilder("sessions")
                        .enforce(MaxRowsPerID(5)) # => sensitivity 5
                        .max("playbackLengthMin", low=0, high=600)
                )
                sensitivity = 5
                return "max_int", query, sensitivity

            return [max_int]

    class Q2Count(BaseApplication):
        def queries(self):
            def count(_session):
                query = (
                    QueryBuilder("sessions")
                    .enforce(MaxRowsPerID(50))
                    .count()
                )
                sensitivity = 50
                return "count_session", query, sensitivity

            return [count]

    class Q3Avg(BaseApplication):
        def queries(self):
            def avg_int(_session):

                query = (
                    QueryBuilder("ratings")
                        .enforce(MaxRowsPerID(20))
                        .average("rating", low=1, high=5)
                )

                # for sum values in range [1, 5], the best is to find the mid point
                # of the range, which is 3 and shift the range to [-2, 2] => sensitivity is 2
                # 2 * 20 = 40

                sensitivity = [40, 20]
                return "mean_rating", query, sensitivity
            return [avg_int]

    class Q4Var(BaseApplication):
        def queries(self):
            def var_float(_session):

                # variance of metric value

                query = (
                    QueryBuilder("metrics")
                        .enforce(MaxRowsPerID(1))
                        .variance("metric1", low=-10, high=100)
                )

                # computes [sum, sum of squares, count]
                #   -> for sum apply the range shift [-10, 100] => [-55, 55] => sensitivity 55
                #   -> for sum squared the range is [0, 10000] => centered is [-4999, 5000] => sensitivity 5000
                sensitivity = [55, 5000, 1]
                return "var_metric1", query, sensitivity

            return [var_float]


    total_rho = 1.0
    total_budget_tmlt = RhoZCDPBudget(total_rho)

    alphas = [1.5, 1.75, 2, 2.5, 3, 4, 5, 6, 8, 16, 32, 64, 1e6]

    total_budget_autodp = [rdp_bank.RDP_zCDP(params={"rho": total_rho}, alpha=alpha) for alpha in alphas]

    scenario.create_session(budget=total_budget_tmlt)

    applications =  [Q1Expo(scenario), Q2Count(scenario), Q3Avg(scenario), Q4Var(scenario)] #[, , Q3Avg(scenario), Q4Var(scenario)]


    history = {}

    def check_in_autodp(mechanisms_autodp, alphas, total_budget_autodp):
        compose = transformer_zoo.Composition()
        mech_autodp_composed = compose(mechanisms_autodp, len(mechanisms_autodp) * [1])
        costs = [mech_autodp_composed.get_RDP(alpha) for alpha in  alphas]

        n_times_possible = max(b/c for c, b in zip(costs, total_budget_autodp, strict=True))

        has_sufficient_budget = any(c < b or math.isclose(c, b)  for c, b in zip(costs, total_budget_autodp, strict=True))

        return has_sufficient_budget, n_times_possible


    for app in applications:

        per_query_budget = RhoZCDPBudget(0.25)

        h = app.execute(scenario.session, [per_query_budget])

        print(f"History={h}")

        for k, v in h.items():
            assert k not in history, "duplicate key"
            history[k] = v

        # compose all mechanisms until now
        mechanisms_autodp = [mech_autodp_composed for mech_autodp_composed, _mech_autodp_amplified_composed, _mechanisms_info in history.values()]
        has_sufficient_budget, n_times_possible = check_in_autodp(mechanisms_autodp, alphas, total_budget_autodp)
        assert has_sufficient_budget, "no alpha found where budget is satisifed"

    assert math.isclose(n_times_possible, 1), "after all of them executed, the overall consumption should match the budget"

    # independent of what mechanism we use, the remaining budget should not be sufficient in autodp


    for app in applications:

        per_query_budget = RhoZCDPBudget(0.05)

        with pytest.raises(InsufficientBudgetException) as exc_info:
            app.execute(scenario.session, [per_query_budget])

        mechanisms_autodp = [mech_autodp_composed for mech_autodp_composed, _mech_autodp_amplified_composed, _mechanisms_info in history.values()]
        mechanisms_autodp.append(exc_info.value.autodp_mechanism[0])
        has_sufficient_budget, n_times_possible = check_in_autodp(mechanisms_autodp, alphas, total_budget_autodp)
        assert not has_sufficient_budget, f"found alpha where budget is satisifed (but should not be)   app={app.__class__.__name__}"