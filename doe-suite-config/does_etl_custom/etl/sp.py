import csv
from doespy.etl.steps.extractors import Extractor
from doespy.etl.steps.transformers import Transformer

from typing import List, Dict

import warnings

import os
from multiprocessing import Pool, cpu_count

import tqdm

import pandas as pd
import json
import itertools

import ast
import numpy as np

from dataclasses import dataclass, field



def _get_directories(exp_result_dir):

    def _list_dir_only(path):
        lst = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
        return lst

    directories = []

    runs = _list_dir_only(exp_result_dir)
    for run in runs:
        run_dir = f"{exp_result_dir}/{run}"
        reps = _list_dir_only(run_dir)
        for rep in reps:
            rep_dir = f"{run_dir}/{rep}"
            hosts = _list_dir_only(rep_dir)
        for host in hosts:
            host_dir = f"{rep_dir}/{host}"
            host_idxs = _list_dir_only(host_dir)
            for host_idx in host_idxs:
                data_dir = f"{host_dir}/{host_idx}"
                directories.append(data_dir)

    return sorted(directories)





@dataclass(eq=True, frozen=True)
class Key:
    round_id: int
    allocation_status: str
    mech_name: str
    cost_name: str
    sampling_name: str
    utility_name: str

@dataclass
class Value:
    request_profit: int = 0
    request_count: int = 0
    request_ids: List[int] = field(default_factory=list)
    request_n_virtual_blocks_distribution: List[int] = field(default_factory=list)


def update_stats(row, req, stats):

    round_id = row["round"]
    alloc_status = row["allocation_status"]

    key = Key(
            round_id = round_id,
            allocation_status = alloc_status,
            mech_name = req["request_info"]["mechanism"]["mechanism"]["name"],
            cost_name = req["request_info"]["cost_original"]["name"],
            sampling_name = req["request_info"]["sampling_info"]["name"],
            utility_name = req["request_info"]["utility_info"]["name"]
        )

    val = stats[key]
    val.request_profit += req["profit"]
    val.request_count += 1
    val.request_ids.append(req["request_id"])
    val.request_n_virtual_blocks_distribution.append(req["request_info"]["selection"]["n_virtual_blocks"])




def build_round_request_summary(run_dir):

    if os.path.exists(f"{run_dir}/round_request_summary.csv"):
        print("Round Request Summary already exists => skipping")
        return

    # load round log
    try:
        df_round_log = pd.read_csv(f"{run_dir}/round_log.csv")
    except pd.errors.EmptyDataError:
        warnings.warn(f"round_log.csv is empty: {run_dir}")
        return
    df_round_log = df_round_log.filter(items=['round', 'allocation_status', 'newly_available_requests', 'newly_accepted_requests'])
    df_round_log.loc[:, "newly_available_requests"] = df_round_log["newly_available_requests"].apply(json.loads)
    df_round_log.loc[:, "newly_accepted_requests"] = df_round_log["newly_accepted_requests"].apply(json.loads)

    df_round_log["allocation_status"] = df_round_log["allocation_status"].fillna("Optimal")

    # process requests
    with open(f"{run_dir}/all_requests.json") as f:
        requests = json.load(f)


    # init available stats (empty value for every possible combination)
    available_stats = {}
    accepted_stats = {}
    for _index, row in df_round_log.iterrows():
        round_id = row["round"]
        alloc_status = row["allocation_status"]

        utility = requests[0]["request_info"]["utility_info"]["name"]
        for entry in requests[0]["workload_info"]["mechanism_mix"]:
            mech_name = entry["mechanism"]["name"]
            costs =  [x["name"] for x in entry["mechanism"]["cost_calibration"]["distribution"]]
            samplings =  [x["name"] for x in entry["mechanism"]["sampling"]["distribution"]]
            for cost, sampling in itertools.product(costs, samplings):
                k = Key(round_id=round_id, allocation_status=alloc_status, mech_name=mech_name, cost_name=cost, sampling_name=sampling, utility_name=utility)
                available_stats[k] = Value()
                accepted_stats[k] = Value()


    # transform requests to dict
    requests = {r["request_id"]: r for r in requests}

    # go through rounds and create summary based on requests
    for _index, row in df_round_log.iterrows():
        round_id = row["round"]
        alloc_status = row["allocation_status"]

        for rid in row['newly_available_requests']:
            req = requests[rid]
            update_stats(row, req, available_stats)

        for rid in row['newly_accepted_requests']:
            req = requests[rid]
            update_stats(row, req, accepted_stats)


    def todict(k, v):
        return {
            # key
            "round": k.round_id,
            "allocation_status": k.allocation_status,
            "request_info.mechanism.name": k.mech_name,
            "request_info.cost.name": k.cost_name,
            "request_info.sampling.name": k.sampling_name, # maybe
            "request_info.utility.name": k.utility_name, # maybe

            # value
            "profit": v.request_profit,
            "n_requests": v.request_count,
            "request_ids": v.request_ids,
            "request_n_virtual_blocks_distribution": v.request_n_virtual_blocks_distribution,
        }



    # convert to results
    lst = []
    for k, v in available_stats.items():
        d = todict(k, v)
        d["status"] = "all_requests"
        lst.append(d)

    for k, v in accepted_stats.items():
        d = todict(k, v)
        d["status"] = "accepted_requests"
        lst.append(d)
    df = pd.DataFrame(lst)

    # store the result in the folder
    df.to_csv(f"{run_dir}/round_request_summary.csv", index=False)

    #print(f"Done: {run_dir}")


class PreProcessingDummyExtractor(Extractor):

    def default_file_regex():
        return ["stderr.log"]

    def extract(self, path: str, options: Dict) -> List[Dict]:
        base_dir = path
        # go multiple levels back
        for _ in range(3):
            base_dir = os.path.dirname(base_dir)

        if os.path.basename(base_dir) == "rep_0" and os.path.basename(os.path.dirname(base_dir)) == "run_0":
            # we only do work once for the first run first rep
            base_dir = os.path.dirname(os.path.dirname(base_dir))

            directories = _get_directories(base_dir)

            with Pool(processes=cpu_count()) as p:
                #p.map(build_round_request_summary, directories)
                for _ in tqdm.tqdm(p.imap_unordered(build_round_request_summary, directories), total=len(directories)):
                    pass

        return []



class FilterCsvExtractor(Extractor):


    delimiter: str = ","

    has_header: bool = True

    fieldnames: List[str] = None


    config_filter: Dict[str, List[str]] = None


    def default_file_regex():
        return [r".*\.csv$"]


    def extract(self, path: str, options: Dict) -> List[Dict]:
        # load file as csv: by default treats the first line as a header
        #   for each later row, we create a dict and add it to the result list to return

        # skip if config does not match
        config = options["$config_flat$"]
        for key, allowed_values in self.config_filter.items():
            if config[key] not in allowed_values:
                return []

        data = []

        with open(path, "r") as f:

            if self.has_header or self.fieldnames is not None:
                reader = csv.DictReader(f, delimiter=self.delimiter, fieldnames=self.fieldnames)
            else:
                reader = csv.reader(f, delimiter=self.delimiter)
            for row in reader:
                data.append(row)

        return data



# TODO: Later can be deleted once we have the new data
def adapt_allocation_algo(x):
    if x['allocation'].startswith("dpf --weighted-dpf"):
        return "weighted-dpf"
    elif x['allocation'].startswith("dpf"):
        return "dpf"

    if "dpk" in x["allocation"]  and "--kp-solver gurobi" in x["allocation"]:
        return "dpk-gurobi"

    if "dpk" in x["allocation"]  and "--kp-solver fptas" in x["allocation"]:
        return "dpk-fptas"

    if x['allocation'] == "ilp":
        return "ilp"

    return x['allocation']

def preprocess(df: pd.DataFrame) -> pd.DataFrame:

    #print(f"cols={df.columns}")

    df.rename(columns={'workload.name': 'workload_info.name'}, inplace=True)

    # for budget mode fix -> slack is nan -> fill it because otherwise dropped
    if "budget.ModeConfig.Unlocking.slack" in df.columns:
        df['budget.ModeConfig.Unlocking.slack'].fillna('-', inplace=True)

    # whenever composition is block-composition -> then we don't have any pa
    df.loc[df["composition"] == "block-composition", 'workload.pa_mix'] = 'nopa'


    return df



class PerRoundSummaryTransformer(Transformer):

    # additional category columns
    cols: List[str] = []

    def transform(self, df: pd.DataFrame, options: Dict) -> pd.DataFrame:


        df = preprocess(df)


        # The goal is to aggregate per <config> x rounds (across mech_name, cost_name, sampling_name)
        # and in a second stage also aggregate across repetitions and report the mean and std

        job_id_cols = ['suite_name', 'suite_id', 'exp_name', 'run', 'workload.rep']

        plot_id_cols = ['workload_info.name', 'composition', 'workload_profit', 'request_info.utility.name']  # for each existing combination, we create a new plot
        line_id_cols = ['allocation', 'workload_mode', 'budget.mode', 'budget.ModeConfig.Unlocking.slack'] # for each existing combination, we create a new line
        category_cols = plot_id_cols + line_id_cols + self.cols

        data_columns = ["profit", "n_requests"]

        df[data_columns] =  df[data_columns].apply(pd.to_numeric)
        df["round"] = df["round"].apply(pd.to_numeric)

        agg_d = {col: "sum" for col in  data_columns}
        agg_d["round"] = "count"
        agg_d["allocation_status"] = lambda x: str(dict(x.value_counts()))

        df.sort_values(by= job_id_cols + category_cols + ["round"], inplace=True)

        # aggregate across mechanisms and costs and samplings (but not across rounds + status)
        df1 = df.groupby(by=["status"] + job_id_cols + category_cols + ["round"]).agg(agg_d).rename({"round": "group_size"}, axis="columns", errors="raise")


        # pivot status to columns (s.t. we have data_columns per status)
        df1.reset_index("status", inplace=True)
        df1.set_index("allocation_status", inplace=True, append=True)
        df1 = df1.pivot(columns="status", values=data_columns)

        # flatten the multi-index columns
        df1.reset_index(inplace=True)
        df1.columns = map(lambda x: x.strip("_"), df1.columns.to_series().str.join('_'))

        # cumsum within
        #df1_cum = df1.groupby(by=job_id_cols + category_cols).agg({"n_requests_accepted_requests": "cumsum", "profit_accepted_requests": "cumsum"}) #.add_suffix("_cum")

        cum_cols = ["n_requests_accepted_requests", "profit_accepted_requests"]
        df1[[x + "_cum" for x in cum_cols]] = df1.groupby(by=job_id_cols + category_cols)[cum_cols].cumsum()
        sum_cols = ["n_requests_all_requests", "profit_all_requests"]
        df1[[x + "_total" for x in sum_cols]] = df1.groupby(by=job_id_cols + category_cols)[sum_cols].transform('sum')

        # compute the fraction of accepted requests in each round of the available requests
        df1["profit_accepted_fraction_perround"] = df1["profit_accepted_requests"] / df1["profit_all_requests"]
        df1["n_requests_accepted_fraction_perround"] = df1["n_requests_accepted_requests"] / df1["n_requests_all_requests"]

        # for the cumulative version, we give the fraction of so far accepted requests from the toal available across all rounds
        df1["profit_accepted_fraction_cum"] = df1["profit_accepted_requests_cum"] / df1["profit_all_requests_total"]
        df1["n_requests_accepted_fraction_cum"] = df1["n_requests_accepted_requests_cum"] / df1["n_requests_all_requests_total"]

        # aggregate accross repetitions
        agg_d = {
                    "profit_accepted_fraction_perround": ["mean", "std"],
                    "n_requests_accepted_fraction_perround": ["mean", "std"],
                    "profit_accepted_fraction_cum": ["mean", "std"],
                    "n_requests_accepted_fraction_cum": ["mean", "std"],
                    "allocation_status": lambda x: str(dict(x.value_counts()))}

        df1 = df1.groupby(by=category_cols + ["round"]).agg(agg_d)

        df1.reset_index(inplace=True)
        df1.columns = map(lambda x: x.strip("_"), df1.columns.to_series().str.join('_'))

        return df1


class ConcatColumnsTransformer(Transformer):

    dest: str
    src: List[str]
    separator: str = "-"

    def transform(self, df: pd.DataFrame, options: Dict) -> pd.DataFrame:
        df[self.dest] = df[self.src].apply(lambda x: self.separator.join(x.astype(str)), axis=1)
        return df

class RoundRequestSummaryTransformer(Transformer):


    # columns that are going to be passed down
    config_cols: List[str] = ['allocation',
                              'composition',

                              'workload_mode', # upc/poisson
                              'workload_profit', # equal/ncd
                              'workload.mechanism_mix',
                              'workload.sub_mix',
                              'workload.pa_mix',

                              'budget.mode',
                              'budget.ModeConfig.Unlocking.slack'
                              ]


    def transform(self, df: pd.DataFrame, options: Dict) -> pd.DataFrame:

        # takes the round request summary and aggregates it across rounds and brings it in the expected shape

        df = preprocess(df)



        job_id_cols = ['suite_name', 'suite_id', 'exp_name', 'run', 'workload_info.name', 'workload.rep'] + self.config_cols

        category_columns = [
            "request_info.mechanism.name",
            "request_info.cost.name",

            # extended version
            "request_info.sampling.name",
            "request_info.utility.name"
        ]

        data_columns = [
            "profit",
            "n_requests",
        ]

        assert set(category_columns).isdisjoint(set(self.config_cols)), "category_columns should not overlap with config_cols"
        assert set(data_columns).isdisjoint(set(self.config_cols)), "data_columns should not overlap with config_cols"


        df[data_columns] =  df[data_columns].apply(pd.to_numeric)

        agg_d = {col: "sum" for col in  data_columns}
        agg_d["round"] = "count"
        agg_d["allocation_status"] = lambda x: str(dict(x.value_counts()))

        assert set(df["status"].unique()) == {"accepted_requests", "all_requests"}, f"status should be either accepted_requests or all_requests  (actual={df['status'].unique()})"

        # aggregate across rounds
        df_accepted = df[df["status"] == "accepted_requests"].groupby(by=job_id_cols + category_columns).agg(agg_d)
        df_accepted.rename({"round": "n_rounds"}, axis=1, inplace=True)

        df_all = df[df["status"] == "all_requests"].groupby(by=job_id_cols + category_columns).agg(agg_d)
        df_all.rename({"round": "n_rounds"}, axis=1, inplace=True)

        # filter out incomplete runs
        n_rounds = df_all["n_rounds"].max()
        df_incomplete = df_all[df_all["n_rounds"] != n_rounds]
        df_incomplete.reset_index(inplace=True)
        df_incomplete = df_incomplete[job_id_cols].drop_duplicates()
        df_incomplete.reset_index(inplace=True, drop=True)

        print("WARNING: We have incomplete runs => are filtered out:")
        print(df_incomplete)

        df_all = df_all[df_all["n_rounds"] == n_rounds]
        df_accepted = df_accepted[df_accepted["n_rounds"] == n_rounds]

        assert len(df_all.index) == len(df_accepted.index), "should have the same number of rows"

        df1 = df_all.join(df_accepted, how='inner', lsuffix='_all', rsuffix='_accepted')

        assert len(df_all.index) == len(df1.index), "join should not change the number of rows"

        # check that there is no NaNs
        assert df1["profit_all"].isna().sum() == 0, "profit_all should not contain any NaNs"
        assert df1["n_requests_all"].isna().sum() == 0, "n_requests_all should not contain any NaNs"
        assert df1["profit_accepted"].isna().sum() == 0
        assert df1["n_requests_accepted"].isna().sum() == 0

        df1["profit_rejected"] = df1["profit_all"] - df1["profit_accepted"]
        df1["n_requests_rejected"] = df1["n_requests_all"] - df1["n_requests_accepted"]


        # aggregate across non-important categories

        main_category_columns = [
            "request_info.mechanism.name",
            "request_info.cost.name",
        ]

        value_columns = [f"{col}_{x}" for col, x in  itertools.product(data_columns, ["all", "accepted", "rejected"])]
        agg_d = {col : "sum" for col in value_columns}
        df1 = df1.groupby(by=job_id_cols + main_category_columns + ["allocation_status_all"]).agg(agg_d)


        df1.reset_index("request_info.cost.name", inplace=True)
        df_pivot = df1.pivot(columns="request_info.cost.name", values=value_columns)

        # Reset the index to move the multi-index columns back to regular columns
        df_pivot = df_pivot.reset_index()
        df_pivot.columns = map(lambda x: x.strip("_"), df_pivot.columns.to_series().str.join('_'))
        df_pivot = df_pivot.reset_index(drop=True)

        def swap_last_two(s, sep='_'):
            parts = s.split(sep)
            if len(parts) >= 2:
                parts[-1], parts[-2] = parts[-2], parts[-1]
            return sep.join(parts)

        # Rename the columns
        df_pivot.columns = [swap_last_two(col) if col.startswith("profit_") or col.startswith("n_requests_")  else col for col in df_pivot.columns]

        return df_pivot