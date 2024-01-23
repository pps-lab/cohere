import matplotlib.ticker as mtick

import matplotlib.container as mcontainer
import numpy as np

import warnings

import matplotlib.pyplot as plt

from does_etl_custom.plots.config import setup_plt, plt_adjust_edges, get_labels, get_colors_bounds
#from .config import setup_plt, plt_adjust_edges, get_labels

#from .config import get_colors_bounds, get_labels
from does_etl_custom.plots.sprites import make_filled_circle

from typing import Optional

from doespy.etl.steps.loaders import PlotLoader
from doespy.etl.etl_util import escape_tuple_str

import pandas as pd
from typing import Dict, List, Union

class UtilityRoundPlotLoaderSPEmanuel(PlotLoader):

    plot_cols: List[str] = ["workload_info.name"]

    plot_cols_values: Dict[str, List[str]] = {"workload_info.name": ["gm:GM", "ml:SGD-PATE", "basic:GM-LM-RR-LSVT"]}

    col_cols: List[str]
    col_cols_values: Dict[str, List[str]]

    y_col_front: str
    y_col_back: Optional[str]

    row_cols: List[str]
    row_cols_values: Dict[str, List[str]]

    # allocation: List[str] = ['greedy', 'weighted-dpf', 'dpk-gurobi', 'ilp']
    # mechanism_mix: List[str] = ['gm:GM', 'gmlm:GM-LM']
    # pa_mix: List[str] = ['midpa', 'highpa', 'vhighpa']
    #
    # slacks: List[str] = ['0.0', '0.2', '0.4', '1.0', '-']

    show_debug_info: bool = False

    def load(self, df: pd.DataFrame, options: Dict, etl_info: Dict) -> None:

        if df.empty:
            return

        output_dir = self.get_output_dir(etl_info)

        n_rows_intial = len(df)
        plot_cols = [(col, self.plot_cols_values[col]) for col in self.plot_cols]
        row_cols = [(col, self.row_cols_values[col]) for col in self.row_cols]
        col_cols = [(col, self.col_cols_values[col]) for col in self.col_cols]
        for col, allowed in plot_cols + row_cols + col_cols:

            # convert column to string for filtering
            df[col] = df[col].astype(str)

            print(f"Filtering {col} to {allowed}    all={df[col].unique()}")
            # filter out non-relevant results
            df = df[df[col].isin(allowed)]
            # convert to categorical
            df[col] = pd.Categorical(df[col], ordered=True, categories=allowed)

        df.sort_values(by=self.plot_cols + self.row_cols + self.col_cols, inplace=True)
        print(f"Filtered out {n_rows_intial - len(df)} rows (based on plot_cols, row_cols, col_cols)  remaining: {len(df)}")

        for idx, df_plot in df.groupby(self.plot_cols):
            print(f"Creating Workload {idx} plot")

            num_rows = np.prod([len(v) for v in self.row_cols_values.values()])
            # number of columns is cartesian product of dictionary values
            num_cols = np.prod([len(v) for v in self.col_cols_values.values()])

            fig_size = [3.441760066417601, 2.38667729342697]
            plt_params = {
                'backend': 'ps',
                'axes.labelsize': 18,
                'legend.fontsize': 12,
                'xtick.labelsize': 16,
                'ytick.labelsize': 16,
                'font.size': 14,
                'figure.figsize': fig_size,
                'font.family': 'Times New Roman',
                'lines.markersize': 8
            }

            plt.rcParams.update(plt_params)
            plt.rc('pdf', fonttype=42)  # IMPORTANT to get rid of Type 3

            figsize = [1.1 * 2 * 3.44, 0.9 * 2.38]

            # plt_adjust_edges(fig)

            # obtain row
            for row_counter, (idx_row, df_row) in enumerate(df_plot.groupby(self.row_cols)):

                fig, ax = plt.subplots(1, num_cols, figsize=figsize)
                if self.show_debug_info:
                    fig.suptitle(f"{escape_tuple_str(idx)}")
                fig.subplots_adjust(left=0.1, right=.99, top=0.87, bottom=0.25, wspace=0.1)

                for col_counter, (idx_col, df_col) in enumerate(df_row.groupby(self.col_cols)):

                    bmode = "Unlocking"
                    y_label = None
                    annotation = None
                    if col_counter == 0: # and row_counter == 0:
                        y_label = "Utility [%]"
                    if col_counter == num_cols - 1: # last col
                        bmode = "Fix"
                        annotation = None


                    show_yticks = col_counter == 0
                    show_legend = row_counter == 0 and col_counter == 0

                    show_xlabel= True #(row_counter == num_rows - 1)
                    show_title = True #(row_counter == 0)

                    make_plot(df_col, ax[col_counter], ycol_front=self.y_col_front, ycol_back=self.y_col_back, title=show_title, budget_mode=bmode,
                              ylabel=y_label, annotation=annotation, show_xlabel=show_xlabel, show_yticks=show_yticks, show_legend=show_legend)

                    # ycol does not end with _cum
                    #ax[col_counter].set_ylim([-0.05, 1.05])

                if row_counter == 0:
                    suffix = "woPA"
                if row_counter == 1:
                    suffix = "wPA"

                #fig.align_ylabels(ax)

                filename = f"workloads_round_{escape_tuple_str(idx)}_{suffix}"
                self.save_plot(fig, filename=filename, output_dir=output_dir, use_tight_layout=False, output_filetypes=["pdf"])
            # plt.close(fig)



# Make a single plot, with composition, allocation, slack, mechanism_mix and pa_mix fixed, returns (min_y, max_y)
# so that sunplots can be scaled
def make_plot(df: pd.DataFrame, ax, budget_mode: str, ycol_front: str, ycol_back, ylabel: Optional[str] = None,
              annotation: Optional[str] = None, title: bool = False, show_xlabel = True, show_yticks = True, show_legend=False) -> None:

    # Filter dataframe based on fixed column values
    df_filtered = df
    if df_filtered.empty:
        warnings.warn("Warning: Dataframe is empty!")
        ax.set_xticks([])
        ax.set_yticks([])
        return

    slack = df_filtered["budget.ModeConfig.Unlocking.slack"].unique()[0]
    assert len(df_filtered["budget.ModeConfig.Unlocking.slack"].unique()) == 1, "This plot is supposed to plot only a single slack value!"

    # Plotting code goes here
    first_round = df_filtered["round"].min()
    df_filtered["round"] = df_filtered["round"] - first_round + 1

    # '#D5E1A3', '#C7B786'
    # "#5dfc00", "#5dfcf7", "#fd9ef7"
    # Define the plot
    ax.errorbar(df_filtered['round'], df_filtered[f'{ycol_front}_mean'],
                yerr=df_filtered[f'{ycol_front}_std'], color="k", linewidth=2, label="Per round")

    if ycol_back is not None:
        # ax.fill_between(df_filtered['round'], df_filtered[f'{ycol_front}_mean'], color='#EEEEEE')
        ax.errorbar(df_filtered['round'], df_filtered[f'{ycol_back}_mean'],
                    yerr=df_filtered[f'{ycol_back}_std'], color='grey', linewidth=2, label="Cumulative")

        # at_round = 35
        # if len(df_filtered[f'{ycol_back}_mean']) > at_round:
        #     plt.annotate('5', xy=(at_round, df_filtered[f'{ycol_back}_mean'].iloc[at_round]), xytext=(-10,10), textcoords='offset points',
        #              arrowprops=dict(arrowstyle="->"), transform=ax.transAxes)

    ax.grid(True, axis="y", linestyle=':', color='0.6', zorder=0, linewidth=1.2)

    xticks = np.arange(0, df_filtered['round'].max() + 1, 20)
    ax.set_xticks(xticks)

    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: f"{(x*100):.0f}"))

    yticks = np.arange(0, 1.0 + 0.1, 0.25)
    ax.set_yticks(yticks)

    ax.set_ylim([-0.05, 1.05])

    if not show_yticks:
        ax.set_yticklabels([])
        ax.tick_params(which='both', width=0, length=0, axis='y')

    # Define labels
    if show_xlabel:
        ax.set_xlabel('Round')
    else:
        ax.set_xticklabels([])

    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title:
        if budget_mode == "Fix":
            ax.set_title('No Unlocking')
        else:
            ax.set_title(f'$\Delta={slack}$')

    if show_legend:
        ax.legend(loc='upper left', bbox_to_anchor=(-0.04, 1.0))

    # Add annotation if provided
    if annotation is not None:
        ax.text(1.05, 0.5, annotation, transform=ax.transAxes, va='center', ha='left')