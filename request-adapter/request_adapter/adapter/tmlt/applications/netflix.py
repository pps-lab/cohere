from datetime import datetime as dt
from typing import Callable, Dict, List, Tuple

import pandas as pd
from pyspark import SparkFiles
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col
from scipy.stats import skewnorm
import os
from tmlt.analytics.binning_spec import BinningSpec
from tmlt.analytics.constraints import (MaxGroupsPerID, MaxRowsPerGroupPerID,
                                        MaxRowsPerID)
from tmlt.analytics.keyset import KeySet
from tmlt.analytics.protected_change import AddRowsWithID
from tmlt.analytics.query_builder import ColumnType, QueryBuilder, QueryExpr
from tmlt.analytics.query_expr import (AverageMechanism,
                                       CountDistinctMechanism, CountMechanism,
                                       StdevMechanism, SumMechanism,
                                       VarianceMechanism)
from tmlt.analytics.session import Session
from request_adapter.adapter.tmlt.applications.base import BaseApplication, BaseScenario


BASE_DATA_DIR = "../data/dummy-netflix"

class NetflixScenario(BaseScenario):

    def init_spark_data(self, spark: SparkSession):



        # load raw data
        spark.sparkContext.addFile(
            os.path.join(BASE_DATA_DIR, "ratings.csv")

        )
        spark.sparkContext.addFile(
            os.path.join(BASE_DATA_DIR, "sessions.csv")
        )
        spark.sparkContext.addFile(
            os.path.join(BASE_DATA_DIR, "shows.csv")
        )
        spark.sparkContext.addFile(
            os.path.join(BASE_DATA_DIR, "users.csv")
        )

        spark.sparkContext.addFile(
            os.path.join(BASE_DATA_DIR, "metrics.csv")
        )

    def init_domains(self) -> Dict:
        return dict() # no domains


    def create_session(self, budget):

        id_space = "privacy_id_space"

        self.session = (
            Session.Builder()
            .with_privacy_budget(budget)
            .with_id_space(id_space)
            .with_private_dataframe(
                "users",
                self.spark.read.csv(SparkFiles.get("users.csv"), header=True, inferSchema=True),
                protected_change=AddRowsWithID(id_column="id", id_space=id_space),
            )
            .with_private_dataframe(
                "ratings",
                self.spark.read.csv(SparkFiles.get("ratings.csv"), header=True, inferSchema=True),
                protected_change=AddRowsWithID(id_column="userId", id_space=id_space),
            )
            .with_private_dataframe(
                "metrics",
                self.spark.read.csv(SparkFiles.get("metrics.csv"), header=True, inferSchema=True),
                protected_change=AddRowsWithID(id_column="userId", id_space=id_space),
            )
            .with_private_dataframe(
                "sessions",
                self.spark.read.csv(SparkFiles.get("sessions.csv"), header=True, inferSchema=True),
                protected_change=AddRowsWithID(id_column="userId", id_space=id_space),
            )
            .with_public_dataframe(
                "shows",
                self.spark.read.csv(SparkFiles.get("shows.csv"), header=True, inferSchema=True)
            )
            .build()
        )

        #return session



    def get_applications(self):

        return [HistogramQuery(self), MinMaxQuery(self), MedianQuery(self), AverageQuery(self), VarianceStdDevQuery(self), QuantileQuery(self), CountQuery(self)]




class HistogramQuery(BaseApplication):

    def queries(self) -> List[Callable[[Dict[str, DataFrame]], Tuple[str, QueryExpr, float]]]:
        def q1(session) -> Tuple[str, QueryExpr, float]:

            # shows per rating

            rating_binspec = BinningSpec(
                bin_edges=[i for i in range(0,6,1)],
                include_both_endpoints=False
            )

            query = (
                QueryBuilder("ratings")
                    .enforce(MaxRowsPerID(50))
                    .histogram("rating", rating_binspec, "rating_binned")
            )

            sensitivity = 50

            return "shows_per_rating", query, sensitivity

        return [q1]



class MinMaxQuery(BaseApplication):

    def queries(self) -> List[Callable[[Dict[str, DataFrame]], Tuple[str, QueryExpr, float]]]:
        def min_int(session) -> Tuple[str, QueryExpr, float]:

            # time when first user joined -> min query of date (int)

            query = (
                QueryBuilder("users")
                    .enforce(MaxRowsPerID(1)) # => sensitivity 1
                    .map(lambda row: {"subscribedOnInt": int(dt.fromisoformat(str(row["subscribedOn"])).timestamp())},
                            new_column_types={"subscribedOnInt": ColumnType.INTEGER},
                            augment=True)
                    .min("subscribedOnInt", low=599616000, high=1609372800)

                    # NOTE: It would be nice to have a way to convert the min back to a date but I would first have to evaluate this in the session and then do the transform in spark
                    #.map(lambda row: {"firstSubscribedOn": dt.fromtimestamp(row["subscribedOnInt_min"]).isoformat()},
                    #        new_column_types={"firstSubscribedOn": ColumnType.VARCHAR},
                    #        augment=True
                    #     )
            )

            sensitivity = 1

            return "first_subscribed_on", query, sensitivity

        def min_float(session) -> Tuple[str, QueryExpr,  float]:

            # min metric value

            query = (
                QueryBuilder("metrics")
                    .enforce(MaxRowsPerID(1)) # sensitivity = 1
                    .min("metric1", low=-10, high=100)
            )

            sensitivity = 1

            return "min_metric1", query, sensitivity


        def max_int(session) -> Tuple[str, QueryExpr, float]:

            # time when last user joined -> max query of date (int)

            query = (
                QueryBuilder("users")
                    .enforce(MaxRowsPerID(1)) # => sensitivity = 1
                    .map(lambda row: {"subscribedOnInt": int(dt.fromisoformat(str(row["subscribedOn"])).timestamp())},
                            new_column_types={"subscribedOnInt": ColumnType.INTEGER},
                            augment=True)
                    .max("subscribedOnInt", low=599616000, high=1609372800)
            )

            sensitivity = 1

            return "last_subscribed_on", query, sensitivity

        def max_float(session) -> Tuple[str, QueryExpr, float]:

            # max metric value

            query = (
                QueryBuilder("metrics")
                    .enforce(MaxRowsPerID(1)) # sensitivity = 1
                    .max("metric1", low=-10, high=100)
            )

            sensitivity = 1

            return "max_metric1", query, sensitivity

        return [min_int, min_float, max_int, max_float]


class MedianQuery(BaseApplication):

    def queries(self) -> List[Callable[[Dict[str, DataFrame]], Tuple[str, QueryExpr, float]]]:
        def median_int(session) -> Tuple[str, QueryExpr, float]:

            # median rating value (per show for group by variant)

            query = (
                QueryBuilder("ratings")
                    .enforce(MaxRowsPerID(20)) # sensitivity = 20
                    .median("rating", low=1, high=5)
            )

            sensitivity = 20

            return "median_rating", query, sensitivity

        def median_float(session) -> Tuple[str, QueryExpr, float]:

            # median metric value

            query = (
                QueryBuilder("metrics")
                    .enforce(MaxRowsPerID(1)) # sensitivity = 1
                    .median("metric1", low=-10, high=100)
            )
            sensitivity = 1
            return "median_metric1", query, sensitivity

        return [median_int, median_float]


class AverageQuery(BaseApplication):

    def queries(self) -> List[Callable[[Dict[str, DataFrame]], Tuple[str, QueryExpr, float]]]:
        def avg_int(session) -> Tuple[str, QueryExpr, float]:

            # avg rating value (per show for group by variant)

            # we could select a specific type of noise here -> GAUSSIAN only works with zCDP

            query = (
                QueryBuilder("ratings")
                    .enforce(MaxRowsPerID(20))
                    .average("rating", low=1, high=5, mechanism=AverageMechanism.DEFAULT)
            )

            # for sum values in range [1, 5], the best is to find the mid point
            # of the range, which is 3 and shift the range to [-2, 2] => sensitivity is 2
            # 2 * 20 = 40

            sensitivity = [40, 20]
            return "mean_rating", query, sensitivity

        def avg_float(session) -> Tuple[str, QueryExpr, float]:

            # median metric value

            query = (
                QueryBuilder("metrics")
                    .enforce(MaxRowsPerID(1))
                    .average("metric1", low=-10, high=100, mechanism=AverageMechanism.DEFAULT)
            )

            # range [-10, 100] => centered around 0: [-55, 55] => sensitivity sum is 55

            sensitivity = [55, 1]

            return "mean_metric1", query, sensitivity

        return [avg_int, avg_float]



class VarianceStdDevQuery(BaseApplication):

    def queries(self) -> List[Callable[[Dict[str, DataFrame]], Tuple[str, QueryExpr, float]]]:
        def var_int(session) -> Tuple[str, QueryExpr, float]:

            # variance of rating value (per show for group by variant)

            query = (
                QueryBuilder("ratings")
                    .enforce(MaxRowsPerID(20))
                    .variance("rating", low=1, high=5, mechanism=VarianceMechanism.DEFAULT)
            )

            # computes [sum, sum of squares, count]
            #   -> for sum apply the range shift [1, 5] => [-2, 2] => sensitivity 2
            #   -> for sum squared the range is [1, 25] => centered is [-12, 12] => sensitivity 12

            sensitivity = [2 * 20, 12 * 20, 20]

            return "var_rating", query, sensitivity

        def var_float(session) -> Tuple[str, QueryExpr, float]:

            # variance of metric value

            query = (
                QueryBuilder("metrics")
                    .enforce(MaxRowsPerID(1))
                    .variance("metric1", low=-10, high=100, mechanism=VarianceMechanism.DEFAULT)
            )

            # computes [sum, sum of squares, count]
            #   -> for sum apply the range shift [-10, 100] => [-55, 55] => sensitivity 55
            #   -> for sum squared the range is [0, 10000] => centered is [-4999, 5000] => sensitivity 5000
            sensitivity = [55, 5000, 1]
            return "var_metric1", query, sensitivity

        def stdev_int(session) -> Tuple[str, QueryExpr, float]:

            # variance of rating value (per show for group by variant)

            query = (
                QueryBuilder("ratings")
                    .enforce(MaxRowsPerID(20))
                    .stdev("rating", low=1, high=5, mechanism=StdevMechanism.DEFAULT)
            )

            # computes [sum, sum of squares, count]
            #   -> for sum apply the range shift [1, 5] => [-2, 2] => sensitivity 2
            #   -> for sum squared the range is [1, 25] => centered is [-12, 12] => sensitivity 12
            sensitivity = [2 * 20, 12 * 20, 20]

            return "stdev_rating", query, sensitivity

        def stdev_float(session) -> Tuple[str, QueryExpr, float]:

            # variance of metric value

            query = (
                QueryBuilder("metrics")
                    .enforce(MaxRowsPerID(1))
                    .stdev("metric1", low=-10, high=100, mechanism=StdevMechanism.DEFAULT)
            )

            # computes [sum, sum of squares, count]
            #   -> for sum apply the range shift [-10, 100] => [-55, 55] => sensitivity 55
            #   -> for sum squared the range is [0, 10000] => centered is [-4999, 5000] => sensitivity 5000 (lower is 0 because could be 0 => 0**2)
            sensitivity = [55, 5000, 1]

            return "stdev_metric1", query, sensitivity

        return [var_float, stdev_int, stdev_float, var_int]



class QuantileQuery(BaseApplication):

    def queries(self) -> List[Callable[[Dict[str, DataFrame]], Tuple[str, QueryExpr, float]]]:
        def quantile_int(session) -> Tuple[str, QueryExpr, float]:

            # quantile rating value (per show for group by variant)
            query = (
                QueryBuilder("ratings")
                    .enforce(MaxRowsPerID(20)) # sensitivity = 20
                    .quantile("rating", 0.75, low=1, high=5) # quantile=0.5 is median
            )

            sensitivity = 20

            return "quantile75_rating", query, sensitivity

        def quantile_float(session) -> Tuple[str, QueryExpr, float]:

            # quantile metric value
            query = (
                QueryBuilder("metrics")
                    .enforce(MaxRowsPerID(1)) # sensitivity = 1
                    .quantile("metric1", 0.25, low=-10, high=100)
            )

            sensitivity = 1

            return "quantile25_metric1", query, sensitivity

        return [quantile_int, quantile_float]


class CountQuery(BaseApplication):

    def queries(self) -> List[Callable[[Dict[str, DataFrame]], Tuple[str, QueryExpr, float]]]:
        def count(session) -> Tuple[str, QueryExpr, float]:

            # number of sessions

            query = (
                QueryBuilder("sessions")
                    .enforce(MaxRowsPerID(50))
                    .count(mechanism=CountMechanism.DEFAULT)
            )

            sensitivity = 50

            return "count_session", query, sensitivity

        def count_distinct_1(session) -> Tuple[str, QueryExpr, float]:

            # number of distinct users that have had a session

            query = (
                QueryBuilder("sessions")
                    .enforce(MaxRowsPerID(1))
                    .count_distinct(columns=["userId"], mechanism=CountDistinctMechanism.DEFAULT)
            )

            sensitivity = 1

            return "count_distinct_session_1", query, sensitivity

        def count_distinct_2(session) -> Tuple[str, QueryExpr]:

            # number of distinct users that have had a session

            query = (
                QueryBuilder("sessions")
                    .enforce(MaxRowsPerID(50))
                    .count_distinct(columns=["userId"], mechanism=CountDistinctMechanism.DEFAULT)
            )

            sensitivity = 1

            return "count_distinct_session_2", query, sensitivity


        return [count, count_distinct_1, count_distinct_2]



def generate_user_metrics():

    df = pd.read_csv(os.path.join(BASE_DATA_DIR, 'users.csv'))

    num_rows = df.shape[0]

    # uses dp_lab way to generate data
    data = skewnorm.rvs(a=5, loc=0, scale=50, size=num_rows)

    df['metric1'] = data
    df["userId"] = df["id"]

    df[["userId", "metric1"]].to_csv(os.path.join(BASE_DATA_DIR, 'metrics.csv'), index=False)
