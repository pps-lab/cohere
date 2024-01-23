
from dash.dependencies import Input, Output
import math
import plotly.express as px


def register_user_arrival_callbacks(app):

    @app.callback(
        Output("input-user-arrival-rate", "value"),
        Output("input-user-daily", "value"),
        Output("input-user-weekly", "value"),
        Output("input-user-annually", "value"),
        Input("slider-user-interarrival-sec", "value"),
    )
    def update_expected_number_of_users(interarrival_sec):
        user_arrival_rate = 1 / interarrival_sec
        daily_users = user_arrival_rate * 3600 * 24
        return (
            user_arrival_rate,
            round(daily_users, 1),
            round(daily_users * 7, 1),
            round(daily_users * 365, 1),
        )


def register_user_distribution_callbacks(app):

    @app.callback(
        Output("input-users-per-group", "value"),
        Output("input-active-users", "value"),
        Input("slider-user-interarrival-sec", "value"),
        Input("slider-allocation-interval-days", "value"),
        Input("slider-K", "value"),
    )
    def update_active_users(user_interarrival_sec, allocation_interval_days, K):
        user_arrival_rate = 1 / user_interarrival_sec
        users_per_group = user_arrival_rate * 3600 * 24 * allocation_interval_days
        active_users = users_per_group * K
        return round(users_per_group, 1), round(active_users, 1)


    @app.callback(
        Output("scatter-plot", "figure"),
        Input("slider-allocation-interval-days", "value"),
        Input("slider-K", "value"),
        Input("slider-T", "value"),
        Input("input-users-per-group", "value"),
        Input("slider-show-n-rounds", "value"),
        Input("show-round-diff", "value"),
    )
    def update_scatter_plot(
        allocation_interval_days, K, T, users_per_group, n_display_rounds, show_round_diff
    ):
        x = users_per_group / T

        start = [x * i for i in range(1, K + 1)]
        end = start[::-1]  # start in reverse order

        total_length = T + K - 1
        mid = (total_length - len(start) - len(end)) * [K * x]

        scatter_sizes = start + mid + end

        assert math.isclose(
            sum(scatter_sizes), users_per_group * K
        ), f"sum mismatch  actual={sum(scatter_sizes)} expected={users_per_group * K}"

        xs = []
        ys = []
        sizes = []
        colors = []

        annotations = []

        def round_id_to_time(round_id):
            return round_id * allocation_interval_days

        for round_id in range(n_display_rounds):
            ys += [round_id] * total_length
            xs += [
                round_id_to_time(i) for i in range(round_id, round_id - total_length, -1)
            ]
            sizes += scatter_sizes
            colors += ["Active"] * total_length
            assert len(xs) == len(ys) and len(ys) == len(sizes), "size mismatch"

        if show_round_diff:
            for round_id in range(n_display_rounds - 1):
                # show removal -> oldest T xs remove x
                ys += [round_id + 0.45] * T
                xs += [
                    round_id_to_time(i)
                    for i in range(
                        round_id - total_length + 1, round_id - total_length + T + 1, 1
                    )
                ]
                sizes += [x] * T
                colors += ["Retired (-)"] * T

                # show additions
                ys += [round_id + 0.55] * T
                new_xs = [
                    round_id_to_time(i) for i in range(round_id + 1, round_id + 1 - T, -1)
                ]
                xs += new_xs
                sizes += [x] * T
                colors += ["New (+)"] * T

                annotations.append(
                    {
                        "x": round_id_to_time(round_id + 2),
                        "y": round_id + 0.5,
                        "text": "Diff [-/+]",
                        "showarrow": False,
                        "xanchor": "left",
                    }
                )

        # Create scatter plot figure
        fig = px.scatter(
            x=xs,
            y=ys,
            size=sizes,
            title="User Distribution in Allocation Rounds",
            color=colors,
            color_discrete_map={"Active": "blue", "New (+)": "green", "Retired (-)": "red"},
            size_max=20,
            labels={"size": "Users", "x": "Join Time [days]", "y": "Round ID"},
        )

        for a in annotations:
            fig.add_annotation(**a)

        rng = list(range(n_display_rounds))
        x_rng = [round_id_to_time(i) for i in rng]
        fig2 = px.scatter(
            x=x_rng,
            y=rng,
            size_max=10,
            symbol_sequence=["square"],
            color_discrete_sequence=["black"],
        )  # , symbol="square"
        fig.add_trace(fig2.data[0])

        fig.update_xaxes(title="User Joined [Days relative to 1st Round]")
        fig.update_yaxes(title="Round ID")

        fig.update_layout(showlegend=True)
        return fig