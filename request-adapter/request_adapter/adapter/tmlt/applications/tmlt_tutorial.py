import math
from datetime import datetime as dt
from typing import Callable, Dict, List, Tuple

from pyspark import SparkFiles
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col
from tmlt.analytics.binning_spec import BinningSpec
from tmlt.analytics.constraints import (MaxGroupsPerID, MaxRowsPerGroupPerID,
                                        MaxRowsPerID)
from tmlt.analytics.keyset import KeySet
from tmlt.analytics.protected_change import AddRowsWithID
from tmlt.analytics.query_builder import ColumnType, QueryBuilder, QueryExpr

from tmlt.analytics.session import Session
from request_adapter.adapter.tmlt.applications.base import BaseApplication, BaseScenario


class TumultTutorialScenario(BaseScenario):

    def init_spark_data(self, spark: SparkSession):

        # load raw data
        spark.sparkContext.addFile(
            "https://tumult-public.s3.amazonaws.com/library-members.csv"
        )
        spark.sparkContext.addFile(
            "https://tumult-public.s3.amazonaws.com/checkout-logs.csv"
        )
        spark.sparkContext.addFile(
            "https://tumult-public.s3.amazonaws.com/nc-zip-codes.csv"
        )
        spark.sparkContext.addFile(
            "https://tumult-public.s3.amazonaws.com/library_books.csv"
        )


    def init_domains(self) -> Dict:
        # Lookup for the domains to build keysets
        domains = {
            "members": {
                "education_level": {
                    "all": [
                        "up-to-high-school",
                        "high-school-diploma",
                        "bachelors-associate",
                        "masters-degree",
                        "doctorate-professional",
                    ]
                },
                "age": {
                    "all": list(range(0, 100)),
                    "young": list(range(5, 18)),
                    "teen": list(range(13, 22)),
                },
                "binned_age": {
                    # bin edges at [0, 10, 20,...,100]
                    "bin10": BinningSpec(bin_edges = [10*i for i in range(0, 11)]),

                    # bin edges at [0, 20, 40,...,100]
                    "bin20": BinningSpec(bin_edges = [20*i for i in range(0, 6)]),
                },

                "gender": {
                    "all": ["female", "male", "nonbinary", "unspecified"]
                },


            },
            "checkouts": {
                "genre": {  # NOTE: needs flatmap function first applied to the genres column
                    "all": [
                        "Mystery/thriller/crime",
                        "History",
                        "Biographies/memoirs",
                        "Romance",
                        "Cookbooks/food writing",
                        "Science fiction",
                        "Fantasy",
                        "Classics/Literature",
                        "Health/wellness",
                        "Religion/spirituality",
                        "Self-help",
                        "True crime",
                        "Political",
                        "Current affairs",
                        "Graphic novels",
                        "Business",
                        "Poetry",
                        "Westerns",
                    ],
                    "small": [
                        "Mystery/thriller/crime",
                        "History",
                        "Romance",
                        "Fantasy",
                        "Classics/Literature",
                        "Children",
                    ],
                }
            }
        }

        return domains

    def create_session(self, budget):

        id_space = "member_id_space"

        self.session = (
            Session.Builder()
            .with_privacy_budget(budget)
            .with_id_space(id_space)
            .with_private_dataframe(
                "checkouts",
                self.spark.read.csv(SparkFiles.get("checkout-logs.csv"), header=True, inferSchema=True),
                protected_change=AddRowsWithID(id_column="member_id", id_space=id_space),
            )
            .with_private_dataframe(
                "members",
                self.spark.read.csv(SparkFiles.get("library-members.csv"), header=True, inferSchema=True),
                protected_change=AddRowsWithID(id_column="id", id_space=id_space),
            )
            .with_public_dataframe(
                "books",
                self.spark.read.csv(SparkFiles.get("library_books.csv"), header=True, inferSchema=True)
            )
            .with_public_dataframe(
                "zip_codes",
                self.spark.read.csv(SparkFiles.get("nc-zip-codes.csv"), header=True, inferSchema=True)
                    .withColumnRenamed("Zip Code", "zip_code")
                    .withColumn("zip_code", col("zip_code").cast("string"))
                    .fillna(0)
            )
            .build()
        )



    def get_applications(self):
        return [Tutorial1(self), Tutorial2(self), Tutorial3(self), Tutorial4(self), Tutorial5(self), Tutorial7(self)]


class Tutorial1(BaseApplication):

    def queries(self) -> List[Callable[[Dict[str, DataFrame]], Tuple[str, QueryExpr, float]]]:

        def q1(session) -> Tuple[str, QueryExpr, float]:
            count_query = QueryBuilder("members").enforce(MaxRowsPerID(1)).count()
            sensitivity = 1
            return "member_count", count_query, sensitivity

        return [q1]


class Tutorial2(BaseApplication):

    def queries(self) -> List[Callable[[Dict[str, DataFrame]], Tuple[str, QueryExpr, float]]]:

        def q1(session) -> Tuple[str, QueryExpr, float]:
            count_query = QueryBuilder("members").enforce(MaxRowsPerID(1)).filter("age < 18").count()
            sensitivity = 1
            return "member_minor_count", count_query, sensitivity

        def q2(session) -> Tuple[str, QueryExpr, float]:
            query = (QueryBuilder("members")
                .enforce(MaxRowsPerID(1))
                .filter("education_level IN ('masters-degree', 'doctorate-professional')")
                .count())
            sensitivity = 1
            return "member_high_edu_count", query, sensitivity

        return [q1, q2]



class Tutorial3(BaseApplication):

    def queries(self) -> List[Callable[[Dict[str, DataFrame]], Tuple[str, QueryExpr, float]]]:

        def q1(session) -> Tuple[str, QueryExpr, float]:
            query = QueryBuilder("members").enforce(MaxRowsPerID(1)).average("age", low=0, high=120)
            sensitivity = [60, 1]
            return "member_mean_age", query, sensitivity

        def q2(session) -> Tuple[str, QueryExpr, float]:
            query = QueryBuilder("members").enforce(MaxRowsPerID(1)).sum("books_borrowed", low=0, high=200)
            sensitivity = 200
            return "member_books_borrowed_count", query, sensitivity

        return [q1, q2]



class Tutorial4(BaseApplication):

    def queries(self) -> List[Callable[[Dict[str, DataFrame]], Tuple[str, QueryExpr, float]]]:

        def q1(session) -> Tuple[str, QueryExpr, float]:

            edu_levels = self.scenario.get_keyset("members", ("education_level", "all"))

            query = (
                QueryBuilder("members")
                .enforce(MaxRowsPerID(1))
                .groupby(edu_levels)
                .average("age", low=0, high=120)
            )
            sensitivity = [60, 1]
            return "member_mean_age_by_edu", query, sensitivity

        def q2(session) -> Tuple[str, QueryExpr, float]:

            young_age_keys = self.scenario.get_keyset("members", ("age", "young"))

            query = (
                QueryBuilder("members")
                .enforce(MaxRowsPerID(1))
                .groupby(young_age_keys)
                .count()
            )
            sensitivity = 1

            return "member_count_by_young_age", query, sensitivity


        def q3(session) -> Tuple[str, QueryExpr, float]:
            teen_edu_keys = self.scenario.get_keyset("members", ("age", "teen"), ("education_level", "all"))

            query = (
                QueryBuilder("members")
                .enforce(MaxRowsPerID(1))
                .groupby(teen_edu_keys)
                .count()
            )

            sensitivity = 1

            return "member_count_by_teen_age_edu", query, sensitivity

        return [q1, q2, q3]


class Tutorial5(BaseApplication):

    def queries(self) -> List[Callable[[Dict[str, DataFrame]], Tuple[str, QueryExpr, float]]]:

        def q1(session) -> Tuple[str, QueryExpr, float]:

            def age_joined(row):
                date_joined = row["date_joined"]
                if isinstance(date_joined, str):
                    date_joined = dt.fromisoformat(date_joined)
                age_at_joining = row["age"] - (dt.today().year - date_joined.year)
                return {"age_joined": age_at_joining}

            age_keys = KeySet.from_dict({"age_joined": self.scenario.get_domain("members", "age", "all") })

            query = (
                QueryBuilder("members")
                    .enforce(MaxRowsPerID(1))
                    .map(age_joined, new_column_types={"age_joined": ColumnType.INTEGER}, augment=True)
                    .groupby(age_keys)
                    .count()
            )

            sensitivity = 1

            return "member_by_age_joined", query, sensitivity

        def q2(session) -> Tuple[str, QueryExpr, float]:

            def expand_genre(row):
                return [{"genre": genre} for genre in row["genres"].split(",")]

            genre_keys = self.scenario.get_keyset("checkouts", ("genre", "all"))

            query = (
                QueryBuilder("checkouts")
                    .enforce(MaxRowsPerID(20))
                    .flat_map(
                        expand_genre,
                        new_column_types={"genre": ColumnType.VARCHAR},
                        max_rows=3,
                        augment=True,
                    )
                    .enforce(MaxRowsPerID(3*20))
                    .groupby(genre_keys)
                    .count()
            )

            sensitivity = 60

            return "checkouts_by_genre", query, sensitivity


        def q3(session) -> Tuple[str, QueryExpr, float]:
            binned_age_gender_keys = self.scenario.get_keyset("members", ("binned_age", "bin10"), ("gender", "all"))

            age_binspec = self.scenario.get_binning_spec("members", "binned_age", "bin10")

            query = (
                QueryBuilder("members")
                    .enforce(MaxRowsPerID(1))
                    .bin_column("age", age_binspec, name="binned_age")
                    .groupby(binned_age_gender_keys)
                    .count()
            )

            sensitivity = 1

            return "member_count_by_bin10age_gender", query, sensitivity


        def q4(session) -> Tuple[str, QueryExpr]:
            #teen_edu_keys = get_keyset("members", ("age", "teen"), ("education_level", "all"))

            zip_code_keys = self.scenario.get_keyset("zip_codes", "City")

            query = (
                QueryBuilder("members")
                    .enforce(MaxRowsPerID(1))
                    .join_public(session.public_source_dataframes["zip_codes"])
                    .groupby(zip_code_keys)
                    .count()
            )

            sensitivity = 1

            return "member_count_by_city", query, sensitivity

        return [q1, q2, q3, q4]



class Tutorial6(BaseApplication):

    def queries(self) -> List[Callable[[Dict[str, DataFrame]], Tuple[str, QueryExpr, float]]]:
        def q1(session) -> Tuple[str, QueryExpr, float]:
            # histogram of the number of checkouts per book
            keyset = self.scenario.get_keyset("books", "title", "author", "isbn")
            count_query = (QueryBuilder("checkouts")
                .enforce(MaxRowsPerID(20))
                .groupby(keyset)
                .count()
            )

            sensitivity = 20

            return "checkouts_by_book", count_query, sensitivity

        def q2(session) -> Tuple[str, QueryExpr, float]:

            # how many patrons (customers) have checked out each of our top five books.

            top_five_df = session.public_source_dataframes["checkouts_by_book"].sort("count", ascending=False).limit(5)
            top_five_keyset = KeySet.from_dataframe(
                    top_five_df.select("title", "author", "isbn"),
            )

            query = (
                QueryBuilder("checkouts")
                    .enforce(MaxGroupsPerID("isbn", 5)) # only consider 5 groups because we only have 5 books
                    .enforce(MaxRowsPerGroupPerID("isbn", 1)) # for each group count each patron only once
                    .groupby(top_five_keyset)
                    .count()
                )

            # https://docs.tmlt.dev/analytics/latest/topic-guides/understanding-sensitivity.html

            # in tmlt/analytics/_query_expr_compiler/_measurement_visitor.py
            # 163:        elif output_measure == RhoZCDP():
            # 164:           return math.sqrt(constraints[0].max) * constraints[1].max
            #  constraints[0] is MaxGroupsPerID and constraints[1] is MaxRowsPerGroupPerID

            sensitivity = math.sqrt(5)

            return "unique_checkouts_by_top5book", query, sensitivity


        return [q1, q2]




class Tutorial7(BaseApplication):

    def queries(self) -> List[Callable[[Dict[str, DataFrame]], Tuple[str, QueryExpr, float]]]:
        def q1(session) -> Tuple[str, QueryExpr, float]:

            if "checkouts_single_genre" in session.private_sources:
                session.delete_view("checkouts_single_genre")
            session.create_view(
                QueryBuilder("checkouts")
                    .flat_map(
                        lambda row: [{"genre": genre} for genre in row["genres"].split(",")],
                        new_column_types={"genre": ColumnType.VARCHAR},
                        max_rows=3,
                        augment=True,
                    ),
                "checkouts_single_genre",
                cache=False,
            )

            if "checkouts_joined" in session.private_sources:
                session.delete_view("checkouts_joined")
            session.create_view(
                QueryBuilder("checkouts_single_genre")
                    .join_private(QueryBuilder("members").rename({"id": "member_id"})),
                "checkouts_joined",
                cache=False,
            )

            binned_age_genre_keys = KeySet.from_dict(
                {
                    "binned_age": self.scenario.get_domain("members", "binned_age", "bin20"),
                    "genre": self.scenario.get_domain("checkouts", "genre", "small")
                }
            )
            n_genres = len(self.scenario.get_domain("checkouts", "genre", "small"))

            age_binspec = self.scenario.get_binning_spec("members", "binned_age", "bin20")

            query = (
                QueryBuilder("checkouts_joined")
                    .bin_column("age", age_binspec, name="binned_age")
                    .enforce(MaxRowsPerID(20))
                    .groupby(binned_age_genre_keys)
                    .count()
                )

            sensitivity = 20

            return "checkouts_by_bin20age_genre", query, sensitivity


        return [q1]