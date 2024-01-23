
from typing import Dict, List, Optional

import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns


from doespy.etl.steps.loaders import PlotLoader





class PerformanceV2PlotLoaderSP(PlotLoader):



    def load(self, df: pd.DataFrame, options: Dict, etl_info: Dict) -> None:


        if df.empty:
            return

        output_dir = self.get_output_dir(etl_info)


        df['mode_composition'] = df['workload_mode'] + "_" + df['composition']

        kinds: list[str] = ['TotalRound']
        group_cols: list[str] = df['mode_composition'].unique().tolist()
        bar_cols: list[str] = ['greedy', 'dpf', 'dpf+', 'weighted-dpf', 'weighted-dpf+', 'dpk-gurobi', 'ilp']
        per_round_or_overall: str = "overall"

        df_filtered = df[df['kind'].isin(kinds) &
                        df['mode_composition'].isin(group_cols) &
                        df['allocation'].isin(bar_cols)].copy()

        job_id_cols = ['suite_name', 'suite_id', 'exp_name', 'run']

        df_filtered['measurement_millis'] = df_filtered['measurement_millis'].astype(int)

        if per_round_or_overall == "overall":
            df_grouped = df_filtered.groupby(['mode_composition', 'allocation'] + job_id_cols + ["workload.rep"]).sum(
                numeric_only=True)
        else:
            df_grouped = df_filtered.groupby(
                ['mode_composition', 'allocation', 'round'] + job_id_cols + ["workload.rep"]).sum(numeric_only=True)

        df_grouped = df_grouped.reset_index()

        palette = {
            'dpf+': 'r',
            'dpf': 'g',
            'ilp': 'b',
            'greedy': 'c',
            'dpk-gurobi': 'm',
            'weighted-dpf': 'y',
            'weighted-dpf+': 'k'
        }

        df_grouped['measurement_secs'] = df_grouped['measurement_millis'] / 1000.


        # FIGURE 1
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))

        g = sns.boxplot(data=df_grouped, ax=ax, x="mode_composition", y='measurement_secs', hue="allocation",
                        order=group_cols, palette=palette, hue_order=bar_cols)
        ax.legend(title="Allocation Methods")
        g.set(ylim=(0, None))
        g.set_ylabel(" + ".join(kinds) + " [sec]")
        g.yaxis.grid(True)
        g.set_axisbelow(True)

        filename = f'mode_comp_{"internal-time_" + "-".join(kinds)}_{"-".join(group_cols)}_{"-".join(bar_cols)}_{per_round_or_overall}'
        self.save_plot(fig, filename=filename, output_dir=output_dir, use_tight_layout=False, output_filetypes=["pdf"])
        plt.close(fig)


        # Figure 2
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))

        g = sns.barplot(data=df_grouped, ax=ax, x="mode_composition", y='measurement_secs', hue="allocation",
                    order=group_cols, palette=palette, hue_order=bar_cols)

        # Calculate the maximum mean value for the measurement_secs
        max_mean_value = df_grouped.groupby(['mode_composition', 'allocation'])['measurement_secs'].mean().max()

        # Annotate each bar with the percentage of the maximum
        for p in g.patches:
            height = p.get_height()
            g.annotate(f'{height:.2f}', (p.get_x() + p.get_width() / 2., p.get_height() + (max_mean_value / 10)),
                    ha='center', va='bottom', fontsize=6)

        ax.legend(title="Allocation Methods")
        g.set_ylabel("Average " + " + ".join(kinds) + " [sec]")
        g.yaxis.grid(True)
        g.set_axisbelow(True)


        filename = f'barplot_mode_comp_{"internal-time_" + "-".join(kinds)}_{"-".join(group_cols)}_{"-".join(bar_cols)}_{per_round_or_overall}'
        self.save_plot(fig, filename=filename, output_dir=output_dir, use_tight_layout=False, output_filetypes=["pdf"])
        plt.close(fig)




        if per_round_or_overall == "overall":
            df_filtered2 = df[
                df['mode_composition'].isin(group_cols) &
                df['allocation'].isin(bar_cols)
                ]
            df_filtered3 = df_filtered2.dropna(subset=["wall_time"]).copy()

            # Prepend '00:' to 'wall_time' if 'hh:' part is missing
            df_filtered3['wall_time'] = df_filtered3['wall_time'].apply(lambda x: '00:' + x if x.count(':') == 1 else x)

            # Convert 'wall_time' to timedelta
            df_filtered3['wall_time'] = pd.to_timedelta(df_filtered3['wall_time'])

            # Convert timedelta to hours
            df_filtered3['wall_time'] = df_filtered3['wall_time'].dt.total_seconds()


            # Figure 3
            fig, ax = plt.subplots(1, 1, figsize=(10, 4))

            g = sns.boxplot(data=df_filtered3, ax=ax, x="mode_composition", y='wall_time', hue="allocation",
                            order=group_cols, palette=palette, hue_order=bar_cols)
            ax.legend(title="Allocation Methods")
            g.set(ylim=(0, None))
            g.set_ylabel("Wall Time [sec]")
            g.yaxis.grid(True)
            g.set_axisbelow(True)

            filename =  f'mode_comp_{"wall_time"}_{"-".join(group_cols)}_{"-".join(bar_cols)}'
            self.save_plot(fig, filename=filename, output_dir=output_dir, use_tight_layout=False, output_filetypes=["pdf"])
            plt.close(fig)

            # Figure 4

            # Create bar plot for the average wall time
            fig, ax = plt.subplots(1, 1, figsize=(10, 4))
            g = sns.barplot(data=df_filtered3, ax=ax, x="mode_composition", y='wall_time', hue="allocation",
                            order=group_cols, palette=palette, hue_order=bar_cols)

            # Calculate the maximum mean value for the wall_time
            max_mean_value_wall = df_filtered3.groupby(['mode_composition', 'allocation'])['wall_time'].mean().max()

            # Annotate each bar with the value
            for p in g.patches:
                height = p.get_height()
                g.annotate(f'{height:.2f}', (p.get_x() + p.get_width() / 2., p.get_height() + (max_mean_value_wall / 10)),
                        ha='center', va='bottom', fontsize=6)

            ax.legend(title="Allocation Methods")
            g.set_ylabel("Average Wall Time [sec]")
            g.yaxis.grid(True)
            g.set_axisbelow(True)

            filename = f'barplot_mode_comp_wall_time_{"-".join(group_cols)}_{"-".join(bar_cols)}'
            self.save_plot(fig, filename=filename, output_dir=output_dir, use_tight_layout=False, output_filetypes=["pdf"])
            plt.close(fig)



            df_filtered3["max_rss_mb"] = df_filtered3["max_rss"] / 1e3

            # Figure 5
            fig, ax = plt.subplots(1, 1, figsize=(10, 4))

            g = sns.boxplot(data=df_filtered3, ax=ax, x="mode_composition", y='max_rss_mb', hue="allocation",
                            order=group_cols, palette=palette, hue_order=bar_cols)
            ax.legend(title="Allocation Methods")
            g.set(ylim=(0, None))
            g.set_ylabel("Max. Memory [MB]")
            g.yaxis.grid(True)
            g.set_axisbelow(True)

            filename = f'mode_comp_{"max_rss"}_{"-".join(group_cols)}_{"-".join(bar_cols)}'
            self.save_plot(fig, filename=filename, output_dir=output_dir, use_tight_layout=False, output_filetypes=["pdf"])
            plt.close(fig)

            # Figure 6
            # Create bar plot for the average max_rss_mb
            fig, ax = plt.subplots(1, 1, figsize=(10, 4))
            g = sns.barplot(data=df_filtered3, ax=ax, x="mode_composition", y='max_rss_mb', hue="allocation",
                            order=group_cols, palette=palette, hue_order=bar_cols)

            # Calculate the maximum mean value for the max_rss_mb
            max_mean_value = df_filtered3.groupby(['mode_composition', 'allocation'])['max_rss_mb'].mean().max()

            # Annotate each bar with the value
            for p in g.patches:
                height = p.get_height()
                g.annotate(f'{height:.2f}', (p.get_x() + p.get_width() / 2., p.get_height() + (max_mean_value / 10)),
                        ha='center', va='bottom', fontsize=6)

            ax.legend(title="Allocation Methods")
            g.set_ylabel("Average Max. Memory [MB]")
            g.yaxis.grid(True)
            g.set_axisbelow(True)

            filename = f'barplot_mode_comp_max_rss_{"-".join(group_cols)}_{"-".join(bar_cols)}'
            self.save_plot(fig, filename=filename, output_dir=output_dir, use_tight_layout=False, output_filetypes=["pdf"])
            plt.close(fig)



class PerformancePlotLoaderSP(PlotLoader):


    def load(self, df: pd.DataFrame, options: Dict, etl_info: Dict) -> None:


        if df.empty:
            return

        output_dir = self.get_output_dir(etl_info)




        # SET PARAMS HERE
        # Is ignored for plots with wall:time and max_rss
        kinds: list[str] = ['TotalRound']
        # The mechanism mixes which should be considered
        group_cols: list[str] = df['workload.mechanism_mix'].unique().tolist()
        # The types of allocations which should be considered
        bar_cols: list[str] = ['greedy', 'dpf', 'dpf+', 'weighted-dpf', 'weighted-dpf+', 'dpk-gurobi', 'ilp']
        # "per_round" or "overall", if "overall" also plots with wall_time and max_rss are generated
        per_round_or_overall: str = "overall"

        # Check params
        for value in group_cols:
            assert value in df["workload.mechanism_mix"].values, \
                f'{value} is not in df["workload.mechanism_mix"]. Available values are: {df["workload.mechanism_mix"].unique()}'
        for value in bar_cols:
            assert value in df["allocation"].values, \
                f'{value} is not in df["allocation"]. Available values are: {df["allocation"].unique()}'
        for value in kinds:
            assert value in df["kind"].values, \
                f'{value} is not in df["kind"]. Available values are: {df["kind"].unique()}'
        assert per_round_or_overall in ["per_round", "overall"], \
            f'The value of per_round_or_overall is {per_round_or_overall}, but it should be either "per_round" or "overall"'

        # Filter to selected values above
        df_filtered = df[df['kind'].isin(kinds) &
                        df['workload.mechanism_mix'].isin(group_cols) &
                        df['allocation'].isin(bar_cols)].copy()

        # The identifiers for a single job
        job_id_cols = ['suite_name', 'suite_id', 'exp_name', 'run']

        # group df_filtered and sum up the metrics
        df_filtered['measurement_millis'] = df_filtered['measurement_millis'].astype(int)

        # Group according to the selected params
        if per_round_or_overall == "overall":
            df_grouped = df_filtered.groupby(['workload.mechanism_mix', 'allocation'] + job_id_cols + ["workload.rep"]).sum(
                numeric_only=True)
        else:
            df_grouped = df_filtered.groupby(
                ['workload.mechanism_mix', 'allocation', 'round'] + job_id_cols + ["workload.rep"]).sum(numeric_only=True)

        df_grouped = df_grouped.reset_index()


        # Define the color palette, for the colors of the boxplots
        palette = {
            'dpf+': 'r',
            'dpf': 'g',
            'ilp': 'b',
            'greedy': 'c',
            'dpk-gurobi': 'm',
            'weighted-dpf': 'y',
            'weighted-dpf+': 'k'
        }

        df_grouped['measurement_secs'] = df_grouped['measurement_millis'] / 1000.

        # Create box plot

        # Figure 1
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))

        g = sns.boxplot(data=df_grouped, ax=ax, x="workload.mechanism_mix", y='measurement_secs', hue="allocation",
                        order=group_cols, palette=palette, hue_order=bar_cols)
        plt.legend(title="Allocation Methods")
        g.set(ylim=(0, None))
        g.set_ylabel(" + ".join(kinds) + " [sec]")
        g.yaxis.grid(True)
        g.set_axisbelow(True)

        filename = f'{"internal-time_" + "-".join(kinds)}_{"-".join(group_cols)}_{"-".join(bar_cols)}_{per_round_or_overall}'
        self.save_plot(fig, filename=filename, output_dir=output_dir, use_tight_layout=False,  output_filetypes=["pdf"])
        plt.close(fig)



        # Create bar plot for the average measurement_secs
        # Figure 2
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        g = sns.barplot(data=df_grouped, ax=ax, x="workload.mechanism_mix", y='measurement_secs', hue="allocation",
                        order=group_cols, palette=palette, hue_order=bar_cols)

        # Calculate the maximum mean value for the measurement_secs
        max_mean_value = df_grouped.groupby(['workload.mechanism_mix', 'allocation'])['measurement_secs'].mean().max()

        # Annotate each bar with the percentage of the maximum
        for p in g.patches:
            height = p.get_height()
            g.annotate(f'{height:.2f}', (p.get_x() + p.get_width() / 2., p.get_height() + (max_mean_value / 10)),
                    ha='center', va='bottom', fontsize=5)

        ax.legend(title="Allocation Methods")
        g.set_ylabel("Average " + " + ".join(kinds) + " [sec]")
        g.yaxis.grid(True)
        g.set_axisbelow(True)

        filename = f'{"barplot_internal-time_" + "-".join(kinds)}_{"-".join(group_cols)}_{"-".join(bar_cols)}_{per_round_or_overall}'
        self.save_plot(fig, filename=filename, output_dir=output_dir, use_tight_layout=False, output_filetypes=["pdf"])
        plt.close(fig)


        # if we have overall, also produce plots with external runtime and memory measurements:
        if per_round_or_overall == "overall":
            df_filtered2 = df[
                df['workload.mechanism_mix'].isin(group_cols) &
                df['allocation'].isin(bar_cols)
                ].copy()
            df_filtered3 = df_filtered2.dropna(subset=["wall_time"]).copy()

            # Prepend '00:' to 'wall_time' if 'hh:' part is missing
            df_filtered3['wall_time'] = df_filtered3['wall_time'].apply(lambda x: '00:' + x if x.count(':') == 1 else x)

            # Convert 'wall_time' to timedelta
            df_filtered3['wall_time'] = pd.to_timedelta(df_filtered3['wall_time'])

            # Convert timedelta to hours
            df_filtered3['wall_time'] = df_filtered3['wall_time'].dt.total_seconds()

            # df_filtered3['mm_allocation'] = df_filtered3['workload.mechanism_mix'] + "_" + df_filtered3['allocation']
            # df_filtered3['mm_allocation'] = df_filtered3['mm_allocation'].astype('category')

            # Create box plot for wall time
            # Figure 3
            fig, ax = plt.subplots(1, 1, figsize=(10, 4))
            g = sns.boxplot(data=df_filtered3, ax=ax, x="workload.mechanism_mix", y='wall_time', hue="allocation",
                            order=group_cols, palette=palette, hue_order=bar_cols)
            ax.legend(title="Allocation Methods")
            g.set(ylim=(0, None))
            g.set_ylabel("Wall Time [sec]")
            g.yaxis.grid(True)
            g.set_axisbelow(True)

            filename = f'{"wall_time"}_{"-".join(group_cols)}_{"-".join(bar_cols)}'
            self.save_plot(fig, filename=filename, output_dir=output_dir, use_tight_layout=False, output_filetypes=["pdf"])
            plt.close(fig)

            # Create bar plot for the average wall time
            # Figure 4
            fig, ax = plt.subplots(1, 1, figsize=(10, 4))
            g = sns.barplot(data=df_filtered3, ax=ax, x="workload.mechanism_mix", y='wall_time', hue="allocation",
                            order=group_cols, palette=palette, hue_order=bar_cols)

            # Calculate the maximum mean value for the wall_time
            max_mean_value_wall = df_filtered3.groupby(['workload.mechanism_mix', 'allocation'])['wall_time'].mean().max()

            # Annotate each bar with the value
            for p in g.patches:
                height = p.get_height()
                g.annotate(f'{height:.2f}', (p.get_x() + p.get_width() / 2., p.get_height() + (max_mean_value_wall / 10)),
                        ha='center', va='bottom', fontsize=6)

            ax.legend(title="Allocation Methods")
            g.set_ylabel("Average Wall Time [sec]")
            g.yaxis.grid(True)
            g.set_axisbelow(True)

            filename = f'barplot_wall_time_{"-".join(group_cols)}_{"-".join(bar_cols)}'
            self.save_plot(fig, filename=filename, output_dir=output_dir, use_tight_layout=False, output_filetypes=["pdf"])
            plt.close(fig)


            df_filtered3["max_rss_mb"] = df_filtered3["max_rss"] / 1e3

            # Create box plot for max rss
            # Figure 5
            fig, ax = plt.subplots(1, 1, figsize=(10, 4))
            g = sns.boxplot(data=df_filtered3, ax=ax, x="workload.mechanism_mix", y='max_rss_mb', hue="allocation",
                            order=group_cols, palette=palette, hue_order=bar_cols)
            ax.legend(title="Allocation Methods")
            g.set(ylim=(0, None))
            g.set_ylabel("Max. Memory [MB]")
            g.yaxis.grid(True)
            g.set_axisbelow(True)

            filename = f'{"max_rss"}_{"-".join(group_cols)}_{"-".join(bar_cols)}'
            self.save_plot(fig, filename=filename, output_dir=output_dir, use_tight_layout=False, output_filetypes=["pdf"])
            plt.close(fig)

            # Create bar plot for the average max_rss_mb
            # Figure 6
            fig, ax = plt.subplots(1, 1, figsize=(10, 4))
            g = sns.barplot(data=df_filtered3, ax=ax, x="workload.mechanism_mix", y='max_rss_mb', hue="allocation",
                            order=group_cols, palette=palette, hue_order=bar_cols)

            # Calculate the maximum mean value for the max_rss_mb
            max_mean_value = df_filtered3.groupby(['workload.mechanism_mix', 'allocation'])['max_rss_mb'].mean().max()

            # Annotate each bar with the value
            for p in g.patches:
                height = p.get_height()
                g.annotate(f'{height:.2f}', (p.get_x() + p.get_width() / 2., p.get_height() + (max_mean_value / 10)),
                        ha='center', va='bottom', fontsize=6)

            ax.legend(title="Allocation Methods")
            g.set_ylabel("Average Max. Memory [MB]")
            g.yaxis.grid(True)
            g.set_axisbelow(True)

            filename = f'barplot_max_rss_{"-".join(group_cols)}_{"-".join(bar_cols)}'
            self.save_plot(fig, filename=filename, output_dir=output_dir, use_tight_layout=False, output_filetypes=["pdf"])
            plt.close(fig)