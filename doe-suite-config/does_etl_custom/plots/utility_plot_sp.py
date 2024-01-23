import warnings
from math import ceil

import matplotlib.ticker as mtick
import matplotlib

import matplotlib.container as mcontainer
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

from does_etl_custom.plots.config import setup_plt_4by2, plt_adjust_edges, get_labels, get_colors_bounds
#from .config import setup_plt, plt_adjust_edges, get_labels

#from .config import get_colors_bounds, get_labels
from does_etl_custom.plots.sprites import make_filled_circle


from doespy.etl.steps.loaders import PlotLoader
from doespy.etl.etl_util import save_notion, escape_tuple_str


import pandas as pd
from typing import Dict, List, Optional


class UtilityPlotLoaderSP(PlotLoader):

    colors: List = ['#D5E1A3', '#C7B786', (166 / 255.0, 184 / 255.0, 216 / 255.0), (76 / 255.0, 114 / 255.0, 176 / 255.0), "#5dfc00", "#5dfcf7", "#fd9ef7"]

    scaling_map: Dict[str, float] = {} #{"lineargaussian": 0.8, "basic": 0.4, "ml": 0.1, "mixed": 0.1}

    legend_bbox_to_anchor_map: Dict[str, tuple[float, float]] = {} # {"lineargaussian": (0., 0.295), "basic": (0., 0.295), "ml": (0., 0.295), "mixed": (0., 0.295)}

    scatter: bool = False

    # for each of those combinations of columns, will have a plot
    plot_cols: List[str] = ["workload.mechanism_mix", "workload_profit", "budget.ModeConfig.Unlocking.slack"]

    plot_cols_values: Dict[str, List[str]] = {
        "workload.mechanism_mix": ["gm:GM"],
        "workload_profit": ["equal-ncd"],
        "budget.ModeConfig.Unlocking.slack": ['0.2', '0.4']
    }

    group_cols: List[str] = ["workload_composition_mode"] # for each combination of these clumns, will have a bar group
    group_cols_values: Dict[str, List[str]] = {"workload_composition_mode": ["upc-block-composition", "poisson-block-composition-pa", "poisson-block-composition"]}
    group_cols_indices: Optional[List[int]] = None # manual override of the order of the groups, to overlay groups

    bar_cols: List[str] = ["allocation"] # for each value of these columns, will have a bar in each group
    bar_cols_values: Dict[str, List[str]] = {"allocation": ['greedy', "weighted-dpf+", "dpk-gurobi"]}


    # Dict[col, Dict[value, replacement]]
    label_lookup: Dict[str, Dict[str, str]]=  {
        "allocation": {
            "greedy": "FCFS",
            "weighted-dpf+": "DPF+",
            "dpk-gurobi": "DPK",
            "ilp": "Upper bound (ILP)"
        },
        "workload_composition_mode": {
            "upc-block-composition": "User-block Comp.",
            "poisson-block-composition": "Poisson Sub.",
            "poisson-block-composition-pa": "Poisson Sub."
        }
    }

    upper_bound_value: Optional[Dict[str, str]] = {
        "workload_composition_mode": "poisson-block-composition-pa",
        "allocation": "ilp"
    }

    show_legend_idx: Optional[List[str]] = None

    workloads: List[str] = ["gm:GM", "ml:SGD-PATE", "basic:GM-LM-RR-LSVT"]

    allocation_algorithms = ['dpf', 'greedy', 'ilp', "weighted-dpf", "dpk-gurobi"]

    compositions: List[str] = ['block-composition', 'block-composition-pa']

    request_types: List[str] = ["elephant", "hare", "mice"]
    notion: Optional[Dict[str, str]] = None


    # less important
    show_ylabel_col: Optional[str] = None
    show_ylabel_values: Optional[List[str]] = None

    show_counts: bool = True
    color_piecharts: bool = True
    mark_timelimit: bool = True

    show_debug_info: bool = False


    def load(self, df: pd.DataFrame, options: Dict, etl_info: Dict) -> None:

        if df.empty:
            return

        output_dir = self.get_output_dir(etl_info)


        #df['allocation'] = df.apply(adapt_allocation_algo, axis=1)

        df.rename(columns={"allocation_status_all": "allocation_status"}, inplace=True)

        df.columns = [col.replace("profit_", "request_info.utility_") if col.startswith("profit_")  else col for col in df.columns]

        #df.rename({"workload.name": "workload_info.name"}, axis=1, inplace=True)

        df_filtered = df.copy()

        n_rows_intial = len(df)

        plot_cols = [(col, self.plot_cols_values[col]) for col in self.plot_cols]
        group_cols = [(col, self.group_cols_values[col]) for col in self.group_cols]
        bar_cols = [(col, self.bar_cols_values[col]) for col in self.bar_cols]
        # filter out non-relevant results
        for col, allowed in plot_cols + group_cols + bar_cols:

            # convert column to string for filtering
            df_filtered[col] = df_filtered[col].astype(str)

            print(f"Filtering {col} to {allowed}    all={df[col].unique()}")
            # filter out non-relevant results
            df_filtered = df_filtered[df_filtered[col].isin(allowed)]
            # convert to categorical
            df_filtered[col] = pd.Categorical(df_filtered[col], ordered=True, categories=allowed)
        df_filtered.sort_values(by=self.plot_cols + self.group_cols + self.bar_cols + ["workload.rep"], inplace=True)

        print(f"Filtered out {n_rows_intial - len(df)} rows (based on plot_cols, group_cols, bar_cols)  remaining: {len(df)}")

        if self.upper_bound_value is not None:
            # filter out non-relevant results
            for col, allowed in plot_cols + [(key, [val]) for key, val in self.upper_bound_value.items()]:

                # convert column to string for filtering
                df[col] = df[col].astype(str)

                print(f"Filtering {col} to {allowed}    all={df[col].unique()}")
                # filter out non-relevant results
                df = df[df[col].isin(allowed)]
                # convert to categoricala
                df[col] = pd.Categorical(df[col], ordered=True, categories=allowed)
            df.sort_values(by=self.plot_cols + self.group_cols + self.bar_cols + ["workload.rep"], inplace=True)

            print(f"Found {len(df)} rows in upper bound df!")

        filenames = []
        for idx, df_plot in df_filtered.groupby(self.plot_cols):
            print(f"Creating Workload {idx} plot")

            df_upper = None
            if self.upper_bound_value is not None:
                df_upper = df.copy()
                for key, val in zip(self.plot_cols, idx):
                    df_upper = df_upper[df_upper[key].isin([val])]
                for key, val in self.upper_bound_value.items():
                    df_upper = df_upper[df_upper[key].isin([val])]

            show_ylabel = self.show_ylabel_col is None
            if self.show_ylabel_col is not None:
                # filter based on values
                show_ylabel = idx[self.plot_cols.index(self.show_ylabel_col)] in self.show_ylabel_values

            assert "workload_profit" in self.plot_cols, "workload_profit must be in plot_cols"
            workload_profit_type = idx[self.plot_cols.index("workload_profit")]

            show_legend = True if self.show_legend_idx is None else np.array_equal(idx, self.show_legend_idx)

            fig = build_utility_plot(df_plot,
                                     df_upper,
                                     group_cols=self.group_cols,
                                     bar_cols=self.bar_cols,
                                     info_cols=[],
                                     label_lookup = self.label_lookup,
                                     colors=self.colors,
                                     request_types=self.request_types, ymax=self.scaling_map.get(idx),
                                     as_scatter=self.scatter,
                                     legend_bbox_to_anchor_map=self.legend_bbox_to_anchor_map,
                                     group_cols_indices=self.group_cols_indices,
                                     workload_profit_type=workload_profit_type,
                                     show_ylabel=show_ylabel,
                                     show_counts=self.show_counts, color_piecharts=self.color_piecharts, mark_timelimit=self.mark_timelimit,
                                     show_legend=show_legend,
                                     show_debug_info=self.show_debug_info)

            if self.show_debug_info:
                fig.suptitle(f"{idx}")
            suffix = 'bar'
            filename = f"workloads_utility_{escape_tuple_str(idx)}_{suffix}_legend"
            filenames.append(filename)
            # use_tight_layout=False because it messes up the custom positioning stuff
            self.save_plot(fig, filename=filename, output_dir=output_dir, use_tight_layout=False, output_filetypes=["pdf"])

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



def build_utility_plot(df, df_upper, group_cols, bar_cols, info_cols, colors, label_lookup, request_types, ymax, as_scatter, show_ylabel,
                       legend_bbox_to_anchor_map, group_cols_indices,
                       workload_profit_type,
                       show_legend=True,
                       show_counts=False, color_piecharts=True, mark_timelimit=True, show_debug_info=False):


    setup_plt_4by2()
    fig, ax = plt.subplots(1, 1)
    fig.tight_layout()
    # plt_adjust_edges(fig)

    fig.subplots_adjust(left=0.18, right=1.0, top=0.95 if show_debug_info else 1.0, bottom=0.1)

    df = df.copy()  # we will mutate the df down

    #################
    # Preprocessing #
    #################
    # Compute total utilities accepted and rejected per-row
    df['total_utility_accepted'] = df['request_info.utility_mice_accepted'] + df['request_info.utility_hare_accepted'] + df['request_info.utility_elephant_accepted']
    df['total_utility_rejected'] = df['request_info.utility_mice_rejected'] + df['request_info.utility_hare_rejected'] + df['request_info.utility_elephant_rejected']

    job_id_cols = ['suite_name', 'suite_id', 'exp_name', 'run']

    mean_upper_bound = None
    if df_upper is not None:
        df_upper['total_utility_accepted'] = df_upper['request_info.utility_mice_accepted'] + df_upper['request_info.utility_hare_accepted'] + df_upper['request_info.utility_elephant_accepted']
        df_upper['total_utility_rejected'] = df_upper['request_info.utility_mice_rejected'] + df_upper['request_info.utility_hare_rejected'] + df_upper['request_info.utility_elephant_rejected']
        df_upper_summed_by_rep = df_upper.groupby(group_cols + bar_cols + info_cols + ["workload.rep"]).sum(numeric_only=True)

        assert len(df_upper_summed_by_rep) == 5, "For now, we hardcode to make sure that we have exactly 5 reps"
        df_upper_summed_by_rep['total_utility_fraction'] = df_upper_summed_by_rep['total_utility_accepted'] / (df_upper_summed_by_rep['total_utility_accepted'] + df_upper_summed_by_rep['total_utility_rejected'])
        mean_upper_bound = df_upper_summed_by_rep['total_utility_fraction'].mean()

    # Add up total counts per request type
    for f in request_types:
        # if f'n_requests_{f}_accepted' in df.columns and f'n_requests_{f}_rejected' in df.columns:
        df[f'total_{f}'] = df[f'n_requests_{f}_accepted'] + df[f'n_requests_{f}_rejected']



    df["count"] = 1

    # 1. sum over everything except group_cols x bar_cols x repetitions (should still be within a job)
    summed_by_rep = df.groupby(job_id_cols + group_cols + bar_cols + info_cols + ["workload.rep"]).sum(numeric_only=True)

    # Compute the fraction of total utility over the whole workload, so _AFTER_ we sum over the mechanisms' utilities
    summed_by_rep['total_utility_fraction'] = summed_by_rep['total_utility_accepted'] / (summed_by_rep['total_utility_accepted'] + summed_by_rep['total_utility_rejected'])


    # 2. aggregate across jobs to get mean / std per bar
    grouped_over_reps = summed_by_rep.groupby(by = group_cols + bar_cols)
    #grouped_over_reps = summed_by_rep.groupby(['composition', 'workload_mode', 'allocation'])
    means = grouped_over_reps['total_utility_fraction'].mean()
    stds = grouped_over_reps['total_utility_fraction'].std()

    # move bar_columns from index to columns (i.e., pivot)
    unstack_levels = [-i for i in range(len(bar_cols), 0, -1)]
    means = means.unstack(unstack_levels)
    stds = stds.unstack(unstack_levels)


    ###################################
    # Drawing the utility bar/scatter #
    ###################################
    # Create bar chart
    bar_width = 0.95

    # setup index map
    index_map = {}
    idx = 0

    n_groups = len(means)
    n_bars_per_group = len(means.columns)
    bar_colors = colors[:n_bars_per_group]
    color_positions = n_groups * bar_colors


    for index, _row in means.iterrows():

        if isinstance(index, str):
            index = [index]


        for column in means.columns:
            if isinstance(column, str):
                column = [column]

            k = tuple(tuple(index) + tuple(column))

            index_map[k] = idx
            idx+=1


    def get_bar_index(row):
        key = tuple(row[key] for key in group_cols + bar_cols)
        return index_map[key]


    def get_x_position(row):

        #print(f"get_x_positions row={row}")

        #print(f"get_x_positions key={key}")

        idx = get_bar_index(row)
        return x_positions[idx]

    # legend=False
    yerr = stds.fillna(0)
    # container = means.plot.bar(yerr=yerr, ax=ax, width=bar_width, color=bar_colors)

    # Use matplotlib to plot bars, a group for each index and a bar in each group for each column
    # Get the number of columns
    num_of_cols = len(means.columns)

    # Create an array with the positions of each bar on the x-axis
    bar_l = np.arange(len(means))
    if group_cols_indices is not None:
        bar_l = group_cols_indices
        assert len(group_cols_indices) == len(means), "group_cols_indices must have the same length as the number of groups in the data"


    # Make the bar chart
    for i, col in enumerate(means.columns):
        w = bar_width / len(means) # divide by number of columns
        bar_pos = [j
                   - (w * num_of_cols / 2.) # center around j
                   + (i*w) # increment for each column
                   + (w/2.) # center in column
                   for j in bar_l]

        individual_colors_as_rgba = [mcolors.to_rgba(bar_colors[i], 1.0) for _ in range(len(bar_pos))]

        # If adjacent bar_pos are the same, low alpha of the first bar to make it transparent
        for j in range(len(bar_pos)-1):
            if bar_pos[j] == bar_pos[j+1]:
                individual_colors_as_rgba[j] = lighten_color(individual_colors_as_rgba[j], amount=1.4)

        ax.bar(bar_pos, means[col], width=w, label=col, yerr=yerr[col], color=individual_colors_as_rgba)

    ax.set_xticks(bar_l)
    ax.set_xticklabels(means.index)

    # extract x positions of the bars + add  bar_width/8. to get the position for the circle
    x_positions = [None] * (len(means.columns) * len(means))
    container_id = 0
    for c in ax.containers:
        # I think containers are the individual calls to ax.bar ??
        if isinstance(c, mcontainer.BarContainer):
            for bar_id, rect in enumerate(c.patches):
                # fill x_positions in horizontal absolute order of the bars
                x_positions[bar_id * len(means.columns) + container_id] = rect.get_x() + bar_width/(n_bars_per_group * 2.)
            container_id += 1

    # determine if this x_positions is in a group, and if yes, if it is first or second
    # 0 = no group, 1 = first in group, 2 = second in group
    x_position_in_group = [0] * len(x_positions)
    for i in range(len(bar_pos)-1):
        if bar_pos[i] == bar_pos[i+1]:
            for j in range(len(means)):
                x_position_in_group[(i * len(means.columns)) + j] = 1
                x_position_in_group[((i+1) * len(means.columns)) + j] = 2

    # print(x_position_in_group)

    # Draw upper bonud line
    if mean_upper_bound is not None:
        y_value = mean_upper_bound.item()
        ax.hlines(
            y_value,
            bar_l[0] - (w * num_of_cols / 2.), bar_l[-1] + (w * num_of_cols / 2.),
            color="#666",
            linestyle="dashed",
            label="ilp"
        )
        if show_debug_info:
            ax.text(0.0, y_value, f"{y_value * 100.0:0.2f}", ha='left', va='bottom')

    if as_scatter:

        df_scatter = summed_by_rep.reset_index()


        df_scatter["x"] = df_scatter.apply(get_x_position, axis=1)


        mark_timelimit = False

        if mark_timelimit:
            df_scatter["edgecolors"]   = df_scatter.apply(lambda x: 'red' if x['allocation_status'] == 'TimeLimit' else 'none', axis=1)
        else:
            df_scatter["edgecolors"] = 'none'

        # pick greyscale scatter colors based on repetitoion (but exclude white)
        n_colors = df['workload.rep'].astype(int).max() + 1 + 2 # +2 to avoid white
        #min_val, max_val = 0.3,1.0
        #orig_cmap = plt.cm.get_cmap('Greys')
        #colors = orig_cmap(np.linspace(min_val, max_val, n_colors))
        #cmap = matplotlib.colors.LinearSegmentedColormap.from_list("mycmap", colors)

        #cmap = plt.cm.get_cmap('Greys', n_colors)
        #scatter_colors = [cmap(i) for i in range(2, n_colors)]
        df_scatter["color"] = "grey"

        #df_scatter["color"] = df_scatter.apply(lambda x: scatter_colors[int(x['workload.rep'])], axis=1)

        markers = ["o", "v", "D", "*", "s", "p", "h"]
        df_scatter["marker"] = df_scatter.apply(lambda x: markers[int(x['workload.rep'])], axis=1)

        for marker in df_scatter["marker"].unique():
            df_tmp = df_scatter[df_scatter["marker"] == marker]

            ax.scatter(x=df_tmp["x"], y=df_tmp['total_utility_fraction'], c=df_tmp["color"], s=40, edgecolor=df_tmp["edgecolors"], marker=marker)


    if ymax is None:

        if as_scatter:
            ymax = df_scatter['total_utility_fraction'].max()
        else:
            ymax = means.max().max()


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


    ax.set_xlabel("")
    # y-scale specific limit? Compute based on max?
    labels = get_x_tick_labels(means, label_lookup)

    #print(f"labels={labels}")
    #print(f"x_positions={x_positions}")
    pos = range(len(labels))
    ax.set_xticks(pos, labels=labels) #, rotation=90.0

    # Rotate the tick labels to be horizontal
    ax.tick_params(axis='x', labelrotation=0)

    # Reduce space on both sides of x-axis to allow for more bar space
    ax.set_xlim(min(x_positions)-0.25, max(x_positions)+0.25)

    ticklines_every = 10  # in percent
    ceiled_ymx = ceil(ymax * ticklines_every) / ticklines_every
    targeted_ymax = ceiled_ymx * 1.8
    ax.set_ylim(0, targeted_ymax)

    # Force a draw so our position calculations relative to the output are correct
    fig.canvas.draw()


    ##############################
    # Drawing fairness piecharts #
    ##############################
    # It gets a bit hairy here with positioning calculations, because we have to draw the circles
    # in absolute coordinates rather than data coordinates, because we dont have equal x and y axis, which would cause
    # our circles to be ellipses.

    # Some positioning configs
    pie_pos_y_top = 1 - 0.08
    pie_pos_y_step = 0.125
    pie_radius = 0.2
    pie_label_pos_y_correction = -0.015
    yticks_request_types = []

    # used to display the picharts not over each other so we can see whats going on
    debug_pie_pos_addition_x = 0.0 # 0.1

    # for each request type (mice, hare, elephant) -> prep data + draw piechart
    for category_i, f in enumerate(request_types):
        if f'n_requests_{f}_accepted' not in df.columns or f'n_requests_{f}_rejected' not in df.columns:
            continue
        n_requests_fraction = summed_by_rep[f'n_requests_{f}_accepted'] / summed_by_rep[f'total_{f}']

        n_requests_fraction_grouped = n_requests_fraction.groupby(group_cols + bar_cols).mean().reset_index()

        n_requests_fraction_grouped.fillna(0, inplace=True)

        pie_pos_x_data_list = n_requests_fraction_grouped.apply(get_bar_index, axis=1)

        pie_pos_y_axis = pie_pos_y_top - (category_i * pie_pos_y_step)
        pie_pos_y_display = ax.transAxes.transform((0, pie_pos_y_axis))
        pie_pos_y_data = ax.transData.inverted().transform(pie_pos_y_display)[1]
        yticks_request_types.append(pie_pos_y_data)

        for index_in_df, pie_pos_x_data_i in enumerate(pie_pos_x_data_list):

            group_status = x_position_in_group[pie_pos_x_data_i]

            debug_pie_pos_addition_x_group = debug_pie_pos_addition_x if group_status == 2 else 0.0
            pie_pos_x_data = x_positions[pie_pos_x_data_i]
            pie_pos_display = ax.transData.transform((pie_pos_x_data + debug_pie_pos_addition_x_group, pie_pos_y_data))
            pie_pos_absolute = fig.dpi_scale_trans.inverted().transform(pie_pos_display)


            edgecolor = None

            #'black' if experiments_not_finished['requests_available'][index_in_df].item() else None

            if color_piecharts:
                color = color_positions[pie_pos_x_data_i]
            else:
                color = colors['fairness_pie']

            # The second in the group (non-PA) should not have a background
            no_background = group_status == 2

            fill = n_requests_fraction_grouped[0][index_in_df]
            make_filled_circle(pie_pos_absolute, radius=pie_radius, fill=fill,
                               fill_color=color, edgecolor=edgecolor, ax=ax, transform=fig.dpi_scale_trans,
                               background_color='#EEEEEE', no_background=no_background)
            # we use no_background because background_color=None would mean use a lighter version of fill_color

            if group_status != 2:
                ax.text(pie_pos_absolute[0], pie_pos_absolute[1] + pie_label_pos_y_correction,
                        f"{n_requests_fraction_grouped[0][index_in_df]:.2f}",
                        horizontalalignment='center',
                        verticalalignment='center',
                        fontsize='small',
                        transform=fig.dpi_scale_trans)


            if show_counts and group_status != 2:
                # Not the second in the group (non PA)
                val = grouped_over_reps['total_utility_fraction'].count()[index_in_df]
                ax.text(pie_pos_x_data, 0.05, val,
                        horizontalalignment='center',
                        verticalalignment='center',
                        transform=ax.transData)

    ###############
    # Draw y-axis #
    ###############
    def truncate_request_type(s):
        return s[:4] + '.' if len(s) > 4 else s
    # What complicates things here is that gridlines will be shown by default for all major ticks,
    # but we dont want gridlines for each number.
    # So we mark the places where we want gridlines as major ticks, and mark other numbers (+ request types at the top)
    # as minor ticks. And then we set the major and minor tick sizes to the same size.
    ax.tick_params(which='both', width=1, length=3)

    yticks_utility = np.arange(0.0, ymax + 0.1, 0.1)
    show_gridlines_at = np.array([int(x * 100.0) % ticklines_every == 0 for x in yticks_utility])

    ax.set_yticks(yticks_utility[show_gridlines_at], minor=False)  # gridlines
    ax.set_yticks(yticks_utility[~show_gridlines_at].tolist() + yticks_request_types, minor=True)

    ax.set_yticklabels(yticks_utility[show_gridlines_at], minor=False)
    ax.set_yticklabels([f"{(x*100):.0f}" for x in yticks_utility[~show_gridlines_at].tolist()] +
                       [truncate_request_type(t) for t in request_types], minor=True)
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: f"{(x*100):.0f}"))

    ax.grid(True, axis="y", linestyle=':', color='0.6', zorder=0, linewidth=1.2)

    if show_debug_info:
        for p in ax.patches:
            if hasattr(p, 'get_height') and hasattr(p, 'get_width') and hasattr(p, 'get_x'):
                ax.annotate(f"{p.get_height() * 100.0:0.2f}", (p.get_x() * 1.005 + (p.get_width() / 2), p.get_height() * 1.005), ha='center', va='bottom')

            elif hasattr(p, 'get_x') and hasattr(p, 'get_y'):
                # for lines?
                ax.annotate(f"{p.get_y() * 100.0:0.2f}", (p.get_x() * 1.005, p.get_y() * 1.005), ha='center', va='bottom')



    if show_ylabel:
        workload_profit_label = label_lookup["workload_profit"][workload_profit_type]
        ax.set_ylabel(f"  {workload_profit_label} [%]             Type")


    handles, legend_labels = ax.get_legend_handles_labels()

    # Legend
    if show_legend:
        if len(bar_cols) == 1:
            single_key = bar_cols[0]
            if label_lookup is not None and single_key in label_lookup:
                labels = [label_lookup[single_key][legend_label] for legend_label in legend_labels]
            else:
                labels = legend_labels
            # ax.legend(labels=labels, handles=handles, bbox_to_anchor=legend_bbox_to_anchor_map.get(workload_name), loc=3)
            ax.legend(labels=labels, handles=handles, loc=3, bbox_to_anchor=(0.002, 0.05))
        else:
            warnings.warn("Will not show legend because multiple bar_cols!")

    return fig

def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])