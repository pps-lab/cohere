import dash
from dash import dcc, html
from hyperparam_explorer.hyperparameter.accounting_info import register_accounting_callbacks
from hyperparam_explorer.hyperparameter.budget_info import (
    load_default_mechanisms,
    register_budget_callbacks,
)
from hyperparam_explorer.hyperparameter.user_info import (
    register_user_arrival_callbacks,
    register_user_distribution_callbacks,
)


external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, title="Hyperparameter")

read_only_input_style = {
    "color": "gray",
    "background-color": "#f4f4f4",
    "cursor": "not-allowed",
}

app.layout = html.Div(
    [
        html.H3("Control Panel"),
        # Control Panel
        html.Div(
            [
                # User Arrival Box
                html.Div(
                    [
                        html.H4("User Arrival"),
                        html.Label("User Expected Interarrival Time [sec]"),
                        dcc.Slider(
                            id="slider-user-interarrival-sec",
                            min=1,
                            max=3600,
                            step=1,
                            value=10,
                            tooltip={"placement": "bottom", "always_visible": True},
                            marks=None,
                        ),
                        html.Div(
                            [
                                html.Br(),
                                html.Label("User Arrival Rate [new users/sec]:"),
                                html.Div(
                                    [
                                        dcc.Input(
                                            id="input-user-arrival-rate",
                                            type="number",
                                            value=0,
                                            readOnly=True,
                                            style=read_only_input_style,
                                        ),
                                    ],
                                    style={
                                        "display": "inline-block",
                                        "margin-left": "10px",
                                        "margin-right": "10px",
                                    },
                                ),
                                html.Br(),
                                html.Br(),
                                html.Label("Expected New Users:"),
                                html.Div(
                                    [
                                        html.Label("daily:"),
                                        dcc.Input(
                                            id="input-user-daily",
                                            type="number",
                                            value=0,
                                            readOnly=True,
                                            style=read_only_input_style,
                                        ),
                                    ],
                                    style={
                                        "display": "inline-block",
                                        "margin-left": "10px",
                                        "margin-right": "10px",
                                    },
                                ),
                                html.Div(
                                    [
                                        html.Label("weekly:"),
                                        dcc.Input(
                                            id="input-user-weekly",
                                            type="number",
                                            value=0,
                                            readOnly=True,
                                            style=read_only_input_style,
                                        ),
                                    ],
                                    style={
                                        "display": "inline-block",
                                        "margin-left": "10px",
                                        "margin-right": "10px",
                                    },
                                ),
                                html.Div(
                                    [
                                        html.Label("annually:"),
                                        dcc.Input(
                                            id="input-user-annually",
                                            type="number",
                                            value=0,
                                            readOnly=True,
                                            style=read_only_input_style,
                                        ),
                                    ],
                                    style={
                                        "display": "inline-block",
                                        "margin-left": "10px",
                                        "margin-right": "10px",
                                    },
                                ),
                            ],
                            style={"margin-left": "20px"},
                        )
                        # Add content for User Arrival box here if needed
                    ],
                    style={
                        "flex": "1 0 calc(33.33% - 50px)",
                        "border": "1px solid #ccc",
                        "padding": "10px",
                        "margin": "10px",
                    },
                ),
                # Sliding Window Box
                html.Div(
                    [
                        html.H4("Sliding Window"),
                        html.Label("Allocation Interval [days]"),
                        dcc.Slider(
                            id="slider-allocation-interval-days",
                            min=1,
                            max=180,
                            step=1,
                            value=7,
                            tooltip={"placement": "bottom", "always_visible": True},
                            marks=None,
                        ),
                        html.Label("User To Group Assignment (T)"),
                        dcc.Slider(
                            id="slider-T",
                            min=1,
                            max=1000,
                            step=1,
                            value=104,
                            tooltip={"placement": "bottom", "always_visible": True},
                            marks=None,
                        ),
                        html.Label(
                            "Allocation Rounds per Group / Groups per Active Window (K)"
                        ),
                        dcc.Slider(
                            id="slider-K",
                            min=1,
                            max=100,
                            step=1,
                            value=12,
                            tooltip={"placement": "bottom", "always_visible": True},
                            marks=None,
                        ),
                    ],
                    style={
                        "flex": "1 0 calc(33.33% - 50px)",
                        "border": "1px solid #ccc",
                        "padding": "10px",
                        "margin": "10px",
                    },
                ),
                # Budget Box
                html.Div(
                    [
                        html.H4("Budget"),
                        html.Label("Total Budget:"),
                        html.Div(
                            [
                                html.Label("\u03B5 ="),
                                dcc.Input(
                                    id="input-total-epsilon",
                                    type="number",
                                    value=3.0,
                                    min=0.0,
                                    step=0.1,
                                ),
                            ],
                            style={
                                "display": "inline-block",
                                "margin-left": "10px",
                                "margin-right": "20px",
                            },
                        ),
                        html.Div(
                            [
                                html.Label("\u03B4 ="),
                                dcc.Input(
                                    id="input-total-delta",
                                    type="number",
                                    value=1e-7,
                                    min=0.0,
                                    max=1.0,
                                    step=1e-10,
                                ),
                            ],
                            style={
                                "display": "inline-block",
                                "margin-left": "10px",
                                "margin-right": "20px",
                            },
                        ),
                        html.Br(),
                        html.Br(),
                        html.Label("Budget Unlocking Slack:"),
                        html.Div(
                            [
                                html.Label("\u0394 ="),
                                dcc.Input(
                                    id="input-slack",
                                    type="number",
                                    value=0.4,
                                    min=0.0,
                                    max=1.0,
                                    step=0.01,
                                ),
                            ],
                            style={
                                "display": "inline-block",
                                "margin-left": "10px",
                                "margin-right": "20px",
                            },
                        ),
                    ],
                    style={
                        "flex": "1 0 calc(33.33% - 50px)",
                        "border": "1px solid #ccc",
                        "padding": "10px",
                        "margin": "10px",
                    },
                ),
            ],
            style={"display": "flex", "flex-wrap": "wrap", "margin-bottom": "20px"},
        ),
        html.H3("Cohere: Scenario Info"),
        # Info Boxes
        html.Div(
            [
                # Info Box spanning two columns
                html.Div(
                    [
                        html.H3("User Distribution"),
                        # Add content for Info Box (2 Columns) here if needed
                        html.Label("Expected Number of Users per Group:  (active set of users consists of K groups)"),
                        html.Div(
                            [
                                dcc.Input(
                                    id="input-users-per-group",
                                    type="number",
                                    value=0,
                                    readOnly=True,
                                    style=read_only_input_style,
                                ),
                            ],
                            style={
                                "display": "inline-block",
                                "margin-left": "10px",
                                "margin-right": "20px",
                            },
                        ),
                        html.Br(),
                        html.Br(),
                        html.Label("Expected Number of Active Users (in every Round):"),
                        html.Div(
                            [
                                dcc.Input(
                                    id="input-active-users",
                                    type="number",
                                    value=0,
                                    readOnly=True,
                                    style=read_only_input_style,
                                ),
                            ],
                            style={
                                "display": "inline-block",
                                "margin-left": "10px",
                                "margin-right": "20px",
                            },
                        ),
                        html.Div(
                            [
                                dcc.Graph(id="scatter-plot"),
                                html.Label("Number of Rounds to Show"),
                                dcc.Slider(
                                    id="slider-show-n-rounds",
                                    min=1,
                                    max=1000,
                                    step=1,
                                    value=5,
                                    tooltip={
                                        "placement": "bottom",
                                        "always_visible": True,
                                    },
                                    marks=None,  # {i: str(i) for i in range(0, 3600, 100)}
                                ),
                                dcc.Checklist(
                                    id="show-round-diff",
                                    options=[
                                        {
                                            "label": "Show Difference between Rounds",
                                            "value": "show",
                                        }
                                    ],
                                    value=["show"],
                                ),
                            ]
                        ),
                    ],
                    style={
                        "flex": "2 0 calc(66.66% - 50px)",
                        "border": "1px solid #ccc",
                        "padding": "10px",
                        "margin": "10px",
                    },
                ),
                # Additional Info Box spanning one column
                html.Div(
                    [
                        html.H4("Budget per Allocation Round"),
                        html.Div(
                            [
                                html.Label("ADP \u03B5 per Round:"),
                                html.Div(
                                    [
                                        html.Label("min:"),
                                        dcc.Input(
                                            id="round-epsilon-min",
                                            type="number",
                                            value=0,
                                            readOnly=True,
                                            style=read_only_input_style,
                                        ),
                                    ],
                                    style={
                                        "display": "inline-block",
                                        "margin-left": "10px",
                                        "margin-right": "10px",
                                    },
                                ),
                                html.Div(
                                    [
                                        html.Label("max:"),
                                        dcc.Input(
                                            id="round-epsilon-max",
                                            type="number",
                                            value=0,
                                            readOnly=True,
                                            style=read_only_input_style,
                                        ),
                                    ],
                                    style={
                                        "display": "inline-block",
                                        "margin-left": "10px",
                                        "margin-right": "10px",
                                    },
                                ),
                                html.Br(),
                                html.Br(),
                                html.Label("ADP \u03B4 per Round:"),
                                html.Div(
                                    [
                                        html.Label("min:"),
                                        dcc.Input(
                                            id="round-delta-min",
                                            type="number",
                                            value=0,
                                            readOnly=True,
                                            style=read_only_input_style,
                                        ),
                                    ],
                                    style={
                                        "display": "inline-block",
                                        "margin-left": "10px",
                                        "margin-right": "10px",
                                    },
                                ),
                                html.Div(
                                    [
                                        html.Label("max:"),
                                        dcc.Input(
                                            id="round-delta-max",
                                            type="number",
                                            value=0,
                                            readOnly=True,
                                            style=read_only_input_style,
                                        ),
                                    ],
                                    style={
                                        "display": "inline-block",
                                        "margin-left": "10px",
                                        "margin-right": "10px",
                                    },
                                ),
                            ],
                            style={"margin-left": "20px"},
                        ),
                        html.Hr(),  # Adding a horizontal divider line
                        html.Div(
                            [
                                html.Label(
                                    "Selected Mechanism: Possible Repetition under Budget"
                                ),
                                html.Div(
                                    [
                                        html.Label("w/ per-round min budget:"),
                                        dcc.Input(
                                            id="input-mech-per-round-min",
                                            type="number",
                                            value=0,
                                            readOnly=True,
                                            style=read_only_input_style,
                                        ),
                                    ],
                                    style={
                                        "display": "inline-block",
                                        "margin-left": "10px",
                                        "margin-right": "20px",
                                    },
                                ),
                                html.Div(
                                    [
                                        html.Label("w/ per-round max budget:"),
                                        dcc.Input(
                                            id="input-mech-per-round-max",
                                            type="number",
                                            value=0,
                                            readOnly=True,
                                            style=read_only_input_style,
                                        ),
                                    ],
                                    style={
                                        "display": "inline-block",
                                        "margin-left": "10px",
                                        "margin-right": "20px",
                                    },
                                ),
                                html.Div(
                                    [
                                        html.Label("w/ total budget:"),
                                        dcc.Input(
                                            id="input-mech-total",
                                            type="number",
                                            value=0,
                                            readOnly=True,
                                            style=read_only_input_style,
                                        ),
                                    ],
                                    style={
                                        "display": "inline-block",
                                        "margin-left": "10px",
                                        "margin-right": "20px",
                                    },
                                ),
                                html.Br(),
                                html.Label("Mechanism Selection:"),
                                html.Div(
                                    [
                                        html.Div(id="slider-output"),
                                    ],
                                    style={
                                        "display": "inline-block",
                                        "margin-left": "10px",
                                        "margin-right": "20px",
                                    },
                                ),
                                html.Div(
                                    children=load_default_mechanisms(),
                                    style={"display": "inline-block"},
                                    id="slider-container",
                                ),
                            ],
                            style={"margin-left": "20px"},
                        ),
                    ],
                    style={
                        "flex": "1 0 calc(33.33% - 50px)",
                        "border": "1px solid #ccc",
                        "padding": "10px",
                        "margin": "10px",
                        "overflow-y": "auto",
                    },
                ),
            ],
            style={"display": "flex", "flex-wrap": "wrap"},
        ),


        # Info Boxes
        html.Div(
            [
                # Info Box spanning three columns
                html.Div(
                    [
                        html.H3("Budget Unlocking"),
                        html.Div(
                            [
                                dcc.Graph(id="accounting-plot"),
                                html.Label("Current Round"),
                                dcc.Slider(
                                    id="slider-current-round",
                                    min=1,
                                    max=1000,
                                    step=1,
                                    value=5,
                                    tooltip={
                                        "placement": "bottom",
                                        "always_visible": True,
                                    },
                                    marks=None,
                                ),
                                html.Div(
                                    children=[],
                                    style={"display": "flex", "flexWrap": "wrap"},
                                    id="round-cost-container",
                                ),
                            ]
                        ),
                    ],
                    style={
                        "flex": "3 0 calc(100% - 50px)",
                        "border": "1px solid #ccc",
                        "padding": "10px",
                        "margin": "10px",
                    },
                ),
            ],
            style={"display": "flex", "flex-wrap": "wrap"},
        ),
    ]
)

register_user_arrival_callbacks(app)
register_budget_callbacks(app)
register_user_distribution_callbacks(app)
register_accounting_callbacks(app)

if __name__ == "__main__":
    app.run_server(debug=True)
