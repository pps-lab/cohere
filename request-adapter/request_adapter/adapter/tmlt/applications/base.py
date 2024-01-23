from abc import ABC, abstractmethod
from pyspark.sql import DataFrame
from typing import List, Dict, Tuple, Callable, Union

from tmlt.analytics.session import Session

from autodp import transformer_zoo

from pyspark.sql import SparkSession

from tmlt.analytics._noise_info import _noise_from_measurement, _NoiseMechanism

from tmlt.analytics.privacy_budget import PureDPBudget, ApproxDPBudget, RhoZCDPBudget


from tmlt.analytics.binning_spec import BinningSpec

from tmlt.analytics.query_builder import QueryExpr

from tmlt.analytics.keyset import KeySet

from tmlt.core.utils.parameters import calculate_noise_scale
from tmlt.core.measures import RhoZCDP


import math

from tmlt.core.utils.exact_number import ExactNumber
from request_adapter.adapter.tmlt.converter import convert_to_audodp
from workload_simulator.request_generator.mechanism import DiscreteGaussianMechanism, ExponentialMechanism, GaussianMechanism


class BaseScenario(ABC):

    def __init__(self) -> None:
        super().__init__()
        self.spark = SparkSession.builder.getOrCreate()
        self.init_spark_data(self.spark)

        self.domains = self.init_domains()

        self.session = None


    @abstractmethod
    def init_spark_data(self, spark):
        pass

    @abstractmethod
    def init_domains(self) -> Dict:
        pass

    @abstractmethod
    def create_session(self, budget) -> Session:
        pass


    @abstractmethod
    def get_applications(self):
        pass


    def get_binning_spec(self, table, column, variant="all"):

        if table not in self.domains:
            raise ValueError(f"Unknown Private Table: {table}")
        elif column not in self.domains[table]:
            raise ValueError(f"Domain of Column={column} is Unknown in Private Table={table}")
        elif variant not in self.domains[table][column]:
            raise ValueError(f"Variant={variant} of Domain of Column={column} is Unknown in Table={table}")

        spec = self.domains[table][column][variant]

        if not isinstance(spec, BinningSpec):
            raise ValueError(f"Table={table}  Column={column}  Variant={variant} is not a Binning Spec")

        return spec


    def get_domain(self, table, column, variant="all"):
        """
            returns the domain as a list
        """

        # public tables
        public = self.session.public_source_dataframes
        if table in public:
            return public[table].select(column).distinct().orderBy(column).rdd.map(lambda x: x[0]).collect()

        # private tables
        elif table not in self.domains:
            raise ValueError(f"Unknown Public/Private Table: {table}")
        elif column not in self.domains[table]:
            raise ValueError(f"Domain of Column={column} is Unknown in Private Table={table}")
        elif variant not in self.domains[table][column]:
            raise ValueError(f"Variant={variant} of Domain of Column={column} is Unknown in Table={table}")

        x = self.domains[table][column][variant]

        if isinstance(x, BinningSpec):
            return x.bins()
        else:
            return x


    def get_keyset(self, table, *cols):

        public = self.session.public_source_dataframes

        if table in public:
            return KeySet.from_dataframe(public[table].select(*cols))
        else: # private table => need to use corss product of domains
            d = {}
            for col in cols:

                if isinstance(col, str):
                    col = (col, "all")
                elif isinstance(col, tuple):
                    assert len(col) == 2 and isinstance(col[0], str) and isinstance(col[1], str)
                else:
                    raise ValueError(f"Unknown Column Type: {col}")

                d[col[0]] = self.get_domain(table, col[0], col[1])

            return KeySet.from_dict(d)

TUMULT_INSUFFICIENT_BUDGET_MESSAGE = "Cannot answer query without exceeding the Session privacy budget"


class InsufficientBudgetException(Exception):
    def __init__(self, message, autodp_mechanism):
        super().__init__(message)
        self.autodp_mechanism = autodp_mechanism

class BaseApplication(ABC):


    def __init__(self, scenario: BaseScenario):
        self.scenario = scenario


    @abstractmethod
    def queries(self) -> List[Callable[[Dict[str, DataFrame]], Tuple[str, QueryExpr, float]]]:
        pass


    def execute(self, session, privacy_costs, skip_exec=False, enforce_sensitivity_check=False):

        history = {}

        for query, cost in zip(self.queries(), privacy_costs, strict=True):
            #print(f"Building Query...")
            name, query_expr, expected_sensitivity = query(session)
            print(f"Query: {name}")

            if enforce_sensitivity_check:
                assert expected_sensitivity is not None, f"Expected Sensitivity must be specified for Query={name}"

            mechanism_autodp = convert_to_audodp(session, query_expr=query_expr, budget=cost, expected_sensitivity=expected_sensitivity)

            if not skip_exec:
                print(f"Evaluating Query: {name}...")
                try:
                    result = session.evaluate(query_expr, cost)
                except RuntimeError as e:
                    assert TUMULT_INSUFFICIENT_BUDGET_MESSAGE in str(e), f"Unexpected Error: {e}"
                    raise InsufficientBudgetException(f"Insufficient Budget for Query = {name}", mechanism_autodp) from e

                print("-> Done")

                if name in session.public_sources:
                    print(f"WARNING: Replace Public Source: {name}")
                    del session.public_source_dataframes[name]


                session.add_public_dataframe(name, result)

                print(f"\n\nNew Release: {name}")
                result.show()

            history[name] = mechanism_autodp

        return history