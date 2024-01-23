import itertools
import json
import os
import time
import warnings

import matplotlib.ticker as mtick
import matplotlib

import matplotlib.container as mcontainer
import numpy as np

import matplotlib.pyplot as plt

from does_etl_custom.plots.config import setup_plt_halfcolumn, plt_adjust_edges, get_labels, get_colors_bounds
#from .config import setup_plt, plt_adjust_edges, get_labels

#from .config import get_colors_bounds, get_labels
from does_etl_custom.plots.sprites import make_filled_circle


from doespy.etl.steps.loaders import PlotLoader
from doespy.etl.etl_util import save_notion, escape_tuple_str


from tqdm import tqdm
import requests

import pandas as pd
from typing import Dict, List, Optional


class SubsamplingAdvantagePlotLoaderSP(PlotLoader):


    colors: List = ['#D5E1A3', '#C7B786', (166 / 255.0, 184 / 255.0, 216 / 255.0), (76 / 255.0, 114 / 255.0, 176 / 255.0), "#5dfc00", "#5dfcf7", "#fd9ef7"]


    scaling_map: Dict[str, float] = {} #{"lineargaussian": 0.8, "basic": 0.4, "ml": 0.1, "mixed": 0.1}

    legend_bbox_to_anchor_map: Dict[str, tuple[float, float]] = {} # {"lineargaussian": (0., 0.295), "basic": (0., 0.295), "ml": (0., 0.295), "mixed": (0., 0.295)}

    scatter: bool = False

    # for each of those combinations of columns, will have a plot
    plot_cols: List[str] = ["workload_info.name"]

    plot_cols_values: Dict[str, List[str]] = {"workload_info.name": ["gm:GM", "ml:SGD-PATE", "basic:GM-LM-RR-LSVT"]}

    group_cols: List[str] = ["composition"] # for each combination of these clumns, will have a bar group
    group_cols_values: Dict[str, List[str]] = {"composition": ['block-composition', 'block-composition-pa']}

    line_cols: List[str] = ["allocation"] # for each value of these columns, will have a bar in each group
    line_cols_values: Dict[str, List[str]] = {"allocation": ['dpf', 'greedy', 'ilp', "weighted-dpf", "dpk-gurobi"]}

    linestyle_cols: List[str]# for each value of these columns, will have a bar in each group
    linestyle_cols_values: Dict[str, List[str]]


    # Dict[col, Dict[value, replacement]]
    label_lookup: Dict[str, Dict[str, str]]=  {

    }



    workloads: List[str] = ["gm:GM", "ml:SGD-PATE", "basic:GM-LM-RR-LSVT"]

    allocation_algorithms = ['dpf', 'greedy', 'ilp', "weighted-dpf", "dpk-gurobi"]

    compositions: List[str] = ['block-composition', 'block-composition-pa']

    request_types: List[str] = ["elephant", "hare", "mice"]

    show_ylabel_col: Optional[str] = None
    show_ylabel_values: Optional[List[str]] = None


    show_xlabel_col: Optional[str] = None
    show_xlabel_values: Optional[List[str]] = None


    show_legend_col: Optional[Dict[str, List[str]]] = None


    # less important
    show_counts: bool = True
    color_piecharts: bool = True
    mark_timelimit: bool = True

    show_debug_info: bool = False

    notion: Optional[Dict[str, str]] = None

    def load(self, df: pd.DataFrame, options: Dict, etl_info: Dict) -> None:

        if df.empty:
            return

        output_dir = self.get_output_dir(etl_info)


        #df['allocation'] = df.apply(adapt_allocation_algo, axis=1)

        df.rename(columns={"allocation_status_all": "allocation_status"}, inplace=True)

        df.columns = [col.replace("profit_", "request_info.utility_") if col.startswith("profit_")  else col for col in df.columns]

        #df.rename({"workload.name": "workload_info.name"}, axis=1, inplace=True)

        n_rows_intial = len(df)

        plot_cols = [(col, self.plot_cols_values[col]) for col in self.plot_cols]
        group_cols = [(col, self.group_cols_values[col]) for col in self.group_cols]
        line_cols = [(col, self.line_cols_values[col]) for col in self.line_cols]
        linestyle_cols = [(col, self.linestyle_cols_values[col]) for col in self.linestyle_cols]

        # filter out non-relevant results
        for col, allowed in plot_cols + group_cols + line_cols + linestyle_cols:

            # convert column to string for filtering
            df[col] = df[col].astype(str)

            print(f"Filtering {col} to {allowed}    all={df[col].unique()}")
            # filter out non-relevant results
            df = df[df[col].isin(allowed)]
            # convert to categorical
            df[col] = pd.Categorical(df[col], ordered=True, categories=allowed)

        #df = df.sort_values(["allocation", "composition", 'workload_mode', "suite_name", "suite_id", "exp_name", "run", "workload.rep"])

        df.sort_values(by=self.plot_cols + self.group_cols + self.line_cols + self.linestyle_cols + ["workload.rep"], inplace=True)

        print(f"Filtered out {n_rows_intial - len(df)} rows (based on plot_cols, group_cols, line_cols)  remaining: {len(df)}")

        filenames = []
        for idx, df_plot in df.groupby(self.plot_cols):
            print(f"Creating Workload {idx} plot")

            show_ylabel = self.show_ylabel_col is None
            if self.show_ylabel_col is not None:
                # filter based on values
                show_ylabel = idx[self.plot_cols.index(self.show_ylabel_col)] in self.show_ylabel_values

            show_xlabel = self.show_xlabel_col is None
            if self.show_xlabel_col is not None:
                # filter based on values
                show_xlabel = idx[self.plot_cols.index(self.show_xlabel_col)] in self.show_xlabel_values

            if self.show_legend_col is None:
                show_legend = True
            else:
                show_legend = all(idx[self.plot_cols.index(col)] in values for col, values in self.show_legend_col.items())


            assert "workload_profit" in self.plot_cols, "workload_profit must be in plot_cols"
            workload_profit_type = idx[self.plot_cols.index("workload_profit")]

            fig, df_means = build_utility_plot(df_plot,
                                     group_cols=self.group_cols,
                                     line_cols=self.line_cols,
                                     linestyle_cols=self.linestyle_cols,
                                     info_cols=[],
                                     label_lookup = self.label_lookup,
                                     colors=self.colors,
                                     request_types=self.request_types, ymax=self.scaling_map.get(idx),
                                     as_scatter=self.scatter,
                                     legend_bbox_to_anchor_map=self.legend_bbox_to_anchor_map,
                                     workload_profit_type=workload_profit_type,
                                     show_ylabel=show_ylabel,
                                     show_xlabel=show_xlabel,
                                     show_legend=show_legend,
                                     show_counts=self.show_counts, color_piecharts=self.color_piecharts, mark_timelimit=self.mark_timelimit, show_debug_info=self.show_debug_info)

            if self.show_debug_info:
                fig.suptitle(f"{idx}")

            suffix = 'scatter' if self.scatter else 'bar'
            filename = f"subsampling_{escape_tuple_str(idx)}_{suffix}_legend"
            filenames.append(filename)
            # use_tight_layout=False because it messes up the custom positioning stuff
            self.save_plot(fig, filename=filename, output_dir=output_dir, use_tight_layout=False, output_filetypes=["pdf"])

            # also store data
            df_means.to_csv(os.path.join(output_dir, f"{filename}.csv"))

        if self.notion is not None:
            save_notion(filenames, etl_info, self.notion)

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



def build_utility_plot(df, group_cols, line_cols, linestyle_cols, info_cols, colors, label_lookup, request_types, ymax, as_scatter, show_ylabel, show_xlabel, legend_bbox_to_anchor_map,
                       workload_profit_type, show_legend,
                       show_counts=False, color_piecharts=True, mark_timelimit=True, show_debug_info=False):


    setup_plt_halfcolumn()

    fig_size = [3.441760066417601, 2.38667729342697]
    plt_params = {
        'figure.figsize': [fig_size[0], fig_size[1]*0.9],
    }

    plt.rcParams.update(plt_params)
    fig, ax = plt.subplots(1, 1)

    fig.tight_layout()
    fig.subplots_adjust(left=0.23, right=1.0, top=1.0, bottom=0.3)

    df = df.copy()  # we will mutate the df down

    #################
    # Preprocessing #
    #################
    # Compute total utilities accepted and rejected per-row
    df['total_utility_accepted'] = df['request_info.utility_mice_accepted'] + df['request_info.utility_hare_accepted'] + df['request_info.utility_elephant_accepted']
    df['total_utility_rejected'] = df['request_info.utility_mice_rejected'] + df['request_info.utility_hare_rejected'] + df['request_info.utility_elephant_rejected']

    # Add up total counts per request type
    for f in request_types:
        # if f'n_requests_{f}_accepted' in df.columns and f'n_requests_{f}_rejected' in df.columns:
        df[f'total_{f}'] = df[f'n_requests_{f}_accepted'] + df[f'n_requests_{f}_rejected']



    df["count"] = 1

    job_id_cols = ['suite_name', 'suite_id', 'exp_name', 'run']

    # 1. sum over everything except group_cols x line_cols x repetitions (should still be within a job)
    summed_by_rep = df.groupby(job_id_cols + group_cols + line_cols + linestyle_cols + info_cols + ["workload.rep"]).sum(numeric_only=True)

    # Compute the fraction of total utility over the whole workload, so _AFTER_ we sum over the mechanisms' utilities
    summed_by_rep['total_utility_fraction'] = summed_by_rep['total_utility_accepted'] / (summed_by_rep['total_utility_accepted'] + summed_by_rep['total_utility_rejected'])


    # 2. aggregate across jobs to get mean / std per bar
    grouped_over_reps = summed_by_rep.groupby(by = group_cols + line_cols + linestyle_cols)
    #grouped_over_reps = summed_by_rep.groupby(['composition', 'workload_mode', 'allocation'])
    means = grouped_over_reps['total_utility_fraction'].mean()
    stds = grouped_over_reps['total_utility_fraction'].std()

    # move bar_columns from index to columns (i.e., pivot)
    unstack_levels = [-i for i in range(len(line_cols) + len(linestyle_cols), 0, -1)]
    means = means.unstack(unstack_levels)
    stds = stds.unstack(unstack_levels)

    # setup index map
    index_map = {}
    idx = 0

    bar_width = 0.95

    n_groups = len(means)
    n_lines_per_group = len(means.columns)
    n_linestyles = len(linestyle_cols)
    bar_colors = colors[:n_lines_per_group]
    linestyles = ['-', '--', '-.', ':'][:n_linestyles]


    for index, _row in means.iterrows():

        if isinstance(index, str):
            index = [index]


        for column in means.columns:
            if isinstance(column, str):
                column = [column]

            k = tuple(tuple(index) + tuple(column))

            index_map[k] = idx
            idx+=1


    # legend=False
    container = means.plot.bar(yerr=stds.fillna(0), ax=ax, width=bar_width, color=bar_colors)

    # Plot each combination of line_cols and linestyle_cols as a separate bar
    # Giving each line in linestyle_cols a different linestyle
    # for idx_row, row in means.iterrows():
    #     ax.plot()

    def get_x_tick_labels(means, label_lookup):
        labels = []
        for idx in means.index:

            if isinstance(idx, str):
                idx = [idx]

            parts = []
            for col, value in zip(means.index.names, idx):
                #print(f"  col={col}  value={value}")
                if col in label_lookup and value in label_lookup[col]:
                    parts.append(label_lookup[col][value])
                else:
                    parts.append(value)

            labels.append("\n".join(parts))

        return labels

    if show_xlabel:
        ax.set_xlabel("Users Requested [%]")
    else:
        ax.set_xlabel(None)

    # y-scale specific limit? Compute based on max?
    labels = get_x_tick_labels(means, label_lookup)

    #print(f"labels={labels}")
    #print(f"x_positions={x_positions}")
    pos = range(len(labels))
    ax.set_xticks(pos, labels=labels, rotation=0.0) #, rotation=90.0

    # if show_ylabel:
    workload_profit_label = label_lookup["workload_profit"][workload_profit_type]
    if show_ylabel:
        ax.set_ylabel(workload_profit_label)
        #ax.set_ylabel(f"[%]")

    ax.set_ylim(0.0, 1.05)
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: f"{(x*100):.0f}"))

    ticklines_every = 25  # in percent
    yticks_utility = np.arange(0.0, 1.1, 0.05)
    show_gridlines_at = np.array([int(x * 100.0) % ticklines_every == 0 for x in yticks_utility])
    ax.set_yticks(yticks_utility[show_gridlines_at], minor=False)  # gridlines

    ax.grid(True, axis="y", linestyle=':', color='0.6', zorder=0, linewidth=1.2)

    if show_debug_info:
        for p in ax.patches:
            ax.annotate(f"{p.get_height():0.2f}", (p.get_x() * 1.005 + (p.get_width() / 2), p.get_height() * 1.005))

    # Legend
    handles, legend_labels = ax.get_legend_handles_labels()


    # Legend
    #if not show_ylabel:
    if show_legend:
        # hack to only show legend in right plot
        if len(line_cols) == 1:
            single_key = line_cols[0]
            if label_lookup is not None and single_key in label_lookup:
                labels = [label_lookup[single_key][legend_label] for legend_label in legend_labels]
            else:
                labels = legend_labels
            # ax.legend(labels=labels, handles=handles, bbox_to_anchor=legend_bbox_to_anchor_map.get(workload_name), loc=3)
            ax.legend(labels=labels, handles=handles, loc=1)
        else:
            warnings.warn("Will not show legend because multiple bar_cols!")
    else:
        ax.get_legend().remove()

    means["improvement_factor"] = means["poisson"] / means["upc"]

    return fig, means
