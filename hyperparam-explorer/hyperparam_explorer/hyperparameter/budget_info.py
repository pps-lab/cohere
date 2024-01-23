import dash
import math
from dash import dcc, html
from dash.dependencies import Input, Output, State, ALL

import plotly.express as px
from hyperparam_explorer.hyperparameter.default_mechanisms import default_evaluation_mechanisms
from workload_simulator.report import calc_n_possible, convert_to_rdp
from workload_simulator.request_generator.mechanism import (
    GaussianMechanism,
    LaplaceMechanism,
    MLNoisySGDMechanism,
    MLPateGaussianMechanism,
    RandResponseMechanism,
    SVTLaplaceMechanism,
)

ALPHAS = [1.5, 1.75, 2, 2.5, 3, 4, 5, 6, 8, 16, 32, 64, 1e6]

mechanism_bank = {}

mechanisms = {
    "GaussianMechanism": GaussianMechanism(),
    "LaplaceMechanism": LaplaceMechanism(),
    "SVTLaplaceMechanism": SVTLaplaceMechanism(),
    "RandResponseMechanism": RandResponseMechanism(),
    "MLPateGaussianMechanism": MLPateGaussianMechanism(),
    "MLNoisySGDMechanism": MLNoisySGDMechanism(),
}


def register_budget_callbacks(app):


    @app.callback(
        Output("round-epsilon-min", "value"),
        Output("round-epsilon-max", "value"),
        Output("round-delta-min", "value"),
        Output("round-delta-max", "value"),
        Input("input-total-epsilon", "value"),
        Input("input-total-delta", "value"),
        Input("input-slack", "value"),
        Input("slider-K", "value"),
    )
    def update_adp_budget(total_epsilon, total_delta, slack, K):

        epsilon_min =  ((1 - slack) / K) * total_epsilon
        epsilon_max = ((1 + slack) / K) * total_epsilon

        delta_min = ((1 - slack) / K) * total_delta
        delta_max = ((1 + slack) / K) * total_delta

        return epsilon_min, epsilon_max, delta_min, delta_max


    # Summarize the selected mechanism
    @app.callback(
        Output("slider-output", "children"),
        Input({"type": "slider-mech-count", "index": ALL}, "value"),
        Input({"type": "slider-mech-count-id", "index": ALL}, "children"),
    )
    def handle_mech_count_update(counts, mech_ids):
        labels = []
        for (
            c,
            mech_id,
        ) in zip(counts, mech_ids):
            if c > 0:
                labels.append(f"{c}x {mechid2label(mech_id)}")
        return " + ".join(labels)


    # Update Mechanism per Round
    @app.callback(
        Output("input-mech-per-round-min", "value"),
        Output("input-mech-per-round-max", "value"),
        Output("input-mech-total", "value"),
        Input({"type": "slider-mech-count", "index": ALL}, "value"),
        Input({"type": "slider-mech-count-id", "index": ALL}, "children"),
        Input("input-total-epsilon", "value"),
        Input("input-total-delta", "value"),
        Input("input-slack", "value"),
        Input("slider-K", "value"),
        prevent_initial_call=True,
    )
    def update_round_mech_combi_count(
        mechanisms_count, mechanisms_label, target_epsilon, target_delta, slack, K
    ):
        if sum(mechanisms_count) == 0:
            return -1, -1, -1

        # 1. convert epsilon / delta to rdp budget
        rdp_budget = convert_to_rdp(
            epsilon=target_epsilon, delta=target_delta, alphas=ALPHAS
        )

        # 2. aggregate selected mechanisms costs
        rdp_cost = [0] * len(ALPHAS)
        for count, m in zip(mechanisms_count, mechanisms_label):
            entry = mechanism_bank[m]
            rdp_cost = [c1 + count * c2 for c1, c2 in zip(rdp_cost, entry["rdp_eps_cost"])]

        # 3. calculate number of times the combination can be used
        d = calc_n_possible(
            n_active_rounds=K, slack=slack, rdp_budget=rdp_budget, rdp_cost=rdp_cost
        )
        return d["round_n_possible_min"], d["round_n_possible_max"], d["total_n_possible"]






def add_mechanism(
    mechanism: str,
    target_epsilon: float,
    target_delta: float,
    name: str,
    subsampling_prob: float = 1.0,
):
    entry_id = f"{mechanism}-{target_epsilon}-{target_delta}-{subsampling_prob}"
    if entry_id in mechanism_bank:  #
        print(f"entry {entry_id} already exists")
        return

    mech_autodp, _info, mcost = mechanisms[mechanism].cache_calibrate(
        epsilon=target_epsilon, delta=target_delta, cost_name=name, alphas=ALPHAS
    )

    if subsampling_prob < 1.0:
        mech_autodp, mcost = mechanisms[mechanism].poisson_amplify(
            mech_autodp,
            amplify_poisson_lambda=subsampling_prob,
            cost_name=name,
            alphas=ALPHAS,
            epsilon=target_epsilon,
            delta=target_delta,
        )

    mechanism_bank[entry_id] = {
        "mechanism": mechanism,
        "name": name,
        "target_epsilon": target_epsilon,
        "target_delta": target_delta,
        "subsampling_prob": subsampling_prob,
        "mech_autodp": mech_autodp,
        "rdp_eps_cost": mcost.rdp["Rdp"]["eps_values"],
    }

    print(f"added {entry_id} to mechanism bank")

    # mcost = {"Rdp": {"eps_values": costs}}



def load_default_mechanisms():
    container = []

    # populate mechanism bank
    default_mechanisms = default_evaluation_mechanisms()
    for k, v in default_mechanisms.items():
        mechanism_bank[k] = v

    # populate the slider container
    for entry_id in mechanism_bank.keys():
        add_slider_mech_count(entry_id, container=container)

    return container


def mechid2label(mech_id):
    entry = mechanism_bank[mech_id]
    label = f"{entry['mechanism']}({entry['name']} \u03B5={entry['target_epsilon']}, \u03B4={entry['target_delta']}, subsampling={entry['subsampling_prob']})"
    return label


def add_slider_mech_count(mech_id, container):
    label = mechid2label(mech_id)

    index = len(container)
    container.append(
        html.Div(
            [
                html.Label(label),
                html.Label(
                    mech_id,
                    style={"display": "none"},
                    id={"type": "slider-mech-count-id", "index": index},
                ),
                dcc.Slider(
                    id={"type": "slider-mech-count", "index": index},
                    min=0,
                    max=100,
                    step=1,
                    value=0,
                    marks=None,
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
            ],
            id=f"slider-div-{index}",
        )
    )


# TODO: We could add a button to calibrate and add a new mechanism