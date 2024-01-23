
from dash.dependencies import Input, Output, State, ALL
from dash import dcc, html

from plotly.subplots import make_subplots


from dataclasses import dataclass
import math
import plotly.graph_objects as go

@dataclass
class Block:
    created_rid: int # block id
    retired_rid: int
    allocations: dict # key: rid, cost





def register_accounting_callbacks(app):


    @app.callback(
        Output("slider-current-round", "min"),
        Output("slider-current-round", "max"),
        Input("slider-K", "value"),
    )
    def update_slider_limits(K):

        start = K + K - 2
        end = start + K - 1
        return start, end


    @app.callback(
        Output('round-cost-container', 'children'),
        Input("slider-K", "value"),
        Input("input-total-epsilon", "value"),
    )
    def update_cost_fields(K, total_epsilon):
        # on change of K, need to add / remove cost fields (by default cost is 1/K)
        container = []

        for round_id in range(K-1, K + K - 1 + K  - 1):

            index = len(container)

            container.append(
                html.Div(
                [
                    html.Label(f"c{round_id}"),
                    html.Label(
                        round_id,
                        style={"display": "none"},
                        id={"type": "input-alloc-cost-round-id", "index": index},
                    ),
                    dcc.Input(
                        id={"type": "input-alloc-cost-round", "index": index},
                        type="number",
                        value= total_epsilon / K,
                        min=0,
                        max=total_epsilon / K,
                        step=0.01,
                    ),
                ],
                id=f"slider-div-{index}",
            ))

        return container

    @app.callback(
        Output("slider-current-round", "value"),
        Input("slider-current-round", "value"),
        Input("slider-K", "value"),
    )
    def update_current_round(current_rid, K):

        max_round_id = K + K  - 1 + K - 1

        current_rid = min(max_round_id, current_rid)

        return current_rid


    @app.callback(
        Output("accounting-plot", "figure"),
        #Output("slider-current-round", "value"),
        Output({"type": "input-alloc-cost-round", "index": ALL}, "max"),
        Input("slider-current-round", "value"),
        Input("slider-K", "value"),
        Input("input-total-epsilon", "value"),
        Input("input-slack", "value"),
        Input({"type": "input-alloc-cost-round", "index": ALL}, "value"),
        Input({"type": "input-alloc-cost-round-id", "index": ALL}, "children"),
    )
    def update_accounting_plot(current_rid, K, total_epsilon, slack, costs, costs_round_ids):

        #print(f"costs={costs}    ids={costs_round_ids}")

        max_round_id = K + K  - 1 + K - 1

        current_rid = min(max_round_id, current_rid)


        if len(costs) == 0:
            return go.Figure(), current_rid, []


        bar_width = 0.8

        blocks = []

        # init blocks
        for rid in range(K + K  - 1 + K - 1):
            b = Block(created_rid=rid, retired_rid=rid + K, allocations={})
            blocks.append(b)


        color_map = {}

        colors = get_colors(len(costs))

        alloc_bars = []
        #cost_limits = []



        # range(K-1, K + K  - 1 + K  - 1)
        for rid, cost in zip(costs_round_ids, costs, strict=True): # apply allocations
            #limit = get_cost_limit(rid,  blocks, total_epsilon, slack, K)
            #if limit < cost and not math.isclose(limit, cost):
            #    print(f"WARNING: cost={cost} exceeds the limit={limit}   for round={rid}")
            #cost_limits.append(limit)

            if rid > max_round_id:
                continue


            if rid <= current_rid:
                apply_cost(rid, cost, blocks, total_epsilon, slack, K)
            else:
                apply_cost(rid, 0, blocks, total_epsilon, slack, K)


            label = f"c{rid}"
            color_map[label] = colors[rid-K+1]


            label = f"c{rid}"
            bar = go.Bar(name=label, x=[rid], y=[cost], text=label, width=bar_width, offsetgroup=2, base=[0], marker=dict(color=color_map[label]), showlegend=False)
            alloc_bars.append(bar)


        #for cost, rid in zip(costs, costs_round_ids):
        #    limit = get_cost_limit(rid,  blocks, total_epsilon, slack, K)
        #    if limit < cost:
        #        print(f"WARNING: cost={cost} exceeds the limit={limit}   for round={rid}")
        #    cost_limits.append(limit)
        #print(cost_limits)

        # from round K-1 onward

        allocations = {}
        x = []
        y_avl = []
        y_locked = []

        budget_violated_rids = set()

        for b in blocks:
            block_id = b.created_rid

            for rid, c, in b.allocations.items():
                if rid not in allocations:
                    allocations[rid] = list()
                allocations[rid].append((block_id, c))

            unlocked_budget = get_unlocked_budget(current_rid, b, total_epsilon, slack, K)
            consumed_budget = get_consumed_budget(current_rid, b)

            avl_budget = unlocked_budget - consumed_budget

            if avl_budget < 0 and not math.isclose(avl_budget, 0):
                budget_violated_rids.add(block_id)
                avl_budget = 0

            locked_budget = total_epsilon - unlocked_budget

            x.append(block_id)
            y_avl.append(avl_budget)
            y_locked.append(locked_budget)


        #print(f"y_avl={y_avl}    budget_violated_rids={budget_violated_rids}")

        bars = []

        y_base_d = {}

        for alloc_rid, data in allocations.items():
            label = f"c{alloc_rid}"

            ys = [c for _block_id, c in data]
            assert len(set(ys)) == 1, "cannot have multiple costs in same round"
            xs = []
            y_base = []
            for block_id, c in data:
                base = y_base_d.get(block_id, 0)
                y_base_d[block_id] = base + c

                xs += [block_id]
                y_base += [base]


            bar = go.Bar(name=label, x=xs, y=ys, text=label, width=bar_width, offsetgroup=1, base=y_base, marker=dict(color=color_map[label]), showlegend=False) #
            bars.append(bar)


        y_base = [y_base_d[rid] for rid in x]

        bar = go.Bar(name="Available", x=x, y=y_avl, text=[round(y, 1) for y in y_avl], textposition="auto" ,width=bar_width, marker=dict(color='white'), offsetgroup=1, base=y_base)
        bars.append(bar)

        y_base = [y_base_d[rid] + avl for rid, avl in zip(x, y_avl)]
        bar = go.Bar(name="Locked", x=x, y=y_locked, text=None, width=bar_width, marker=dict(color='black'), offsetgroup=1, base= y_base)  # would work
        bars.append(bar)


        fig = make_subplots(rows=2, cols=1, shared_xaxes=False, row_heights=[0.8, 0.2], subplot_titles=("Budget per Block", "Allocation Cost per Round"))#

        for bar in bars:
            fig.add_trace(bar, row=1, col=1)






        # draw bounding boxes


        x_unlocked = []
        y_unlocked = []
        for b in get_active_blocks(current_rid, blocks):
            left_x = b.created_rid - bar_width / 2
            right_x = b.created_rid + bar_width / 2
            level = get_unlocked_budget(current_rid, b, total_epsilon, slack, K)
            y_unlocked += [level, level, None]

            x_unlocked += K * [left_x, right_x, None]


        # expand
        y_unlocked = int(len(y_unlocked) // 3) * y_unlocked

        fig.add_scatter(x=x_unlocked, y=y_unlocked, line=dict(color='black', dash="dot"), mode="lines", showlegend=False)

        x = []
        y = []
        x_red = []
        y_red = []
        for b in blocks:

            left_x = b.created_rid - bar_width / 2
            right_x = b.created_rid + bar_width / 2

            new_x = [left_x, right_x, right_x, left_x, left_x, None]
            new_y = [0, 0, total_epsilon, total_epsilon, 0, None]

            if b.created_rid in budget_violated_rids:
                x_red += new_x
                y_red += new_y
            else:
                x += new_x
                y += new_y


        fig.add_scatter(x=x, y=y, line=dict(color='black'), mode="lines", showlegend=False)
        fig.add_scatter(x=x_red, y=y_red, line=dict(color='red'), mode="lines", showlegend=False)





        #fig1 = go.Figure(data=alloc_bars)

        min_active_id = min(b.created_rid for b in get_active_blocks(current_rid, blocks))
        max_rid = blocks[-1].created_rid


        #fig1.show()
        for bar in alloc_bars:
            fig.add_trace(bar, row=2, col=1)


        min_cost = (1-slack) / K * total_epsilon
        max_cost = (1+slack) / K * total_epsilon

        scat = go.Scatter(x=[0 - bar_width / 2, max_rid + bar_width / 2, None, 0 - bar_width / 2, max_rid + bar_width / 2], y=[min_cost, min_cost, None, max_cost, max_cost], line=dict(color='black', dash="dot"), mode="lines", showlegend=False)
        fig.add_trace(scat, row=2, col=1)


        # add retired shade
        max_retired_rid = min_active_id - 1
        fig.add_vrect(x0=0 - bar_width / 2, x1=max_retired_rid + bar_width / 2,
                    annotation_text="retired", annotation_position="top left",
                    annotation=dict(font_size=20, font_family="Times New Roman"),
                    fillcolor="grey", opacity=0.50, line_width=0)


        # add future shade

        fig.add_vrect(x0=current_rid + 1 - bar_width / 2, x1=max_rid + bar_width / 2,
                    annotation_text="future", annotation_position="top left",
                    annotation=dict(font_size=20, font_family="Times New Roman"),
                    fillcolor="grey", opacity=0.50, line_width=0)


        fig.update_xaxes(title="Block Id", tickmode='linear', dtick=1, row=1, col=1)  # Set the x-axis range for subplot 1

        fig.update_xaxes(title="Round Id", range=[-0.4, max_rid + bar_width / 2], tickmode='linear', dtick=1, row=2, col=1)  # Set the x-axis range for subplot 1

        fig.update_yaxes(title="Budget [\u03B5]", range=[0, 1.2 * total_epsilon ], row=1, col=1)  # Set the x-axis range for subplot 1



        fig.update_layout(height=int(800))


        cost_limits = len(costs) * [(1+slack) * total_epsilon / K + 0.01]
        return fig, cost_limits





def get_active_blocks(rid, blocks):
    return [b for b in blocks if b.created_rid <= rid and b.retired_rid > rid]


def get_unlocked_budget(rid, block, total_epsilon, slack, K):

    assert block.retired_rid - block.created_rid == K, "error in retired / created rid"


    if rid < block.created_rid:
        return 0
    elif rid >= block.retired_rid:
        return total_epsilon # all budget is unlocked
    else:
        assert rid >= block.created_rid and rid < block.retired_rid # active block

        k = rid - block.created_rid + 1

        assert k >= 1 and k <= K

        s1 = sum(slack for i in range(min(k, K // 2)))
        s2 = sum(-slack for i in range(math.ceil(K / 2) + 1, k+1))
        epsilon_unlocked = total_epsilon / K * (k + s1 + s2)

        return epsilon_unlocked

def get_consumed_budget(rid, block):
    consumed_budget = 0
    for rid2, cost in block.allocations.items():
        if rid2 <= rid:
            consumed_budget += cost
    return consumed_budget


def get_cost_limit(rid,  blocks, total_epsilon, slack, K):
    limits = []
    for b in get_active_blocks(rid, blocks):
        consumed_budget = get_consumed_budget(rid, b)
        unlocked_budget = get_unlocked_budget(rid, b, total_epsilon, slack, K)

        assert unlocked_budget >= consumed_budget

        limits += [unlocked_budget - consumed_budget]

    return min(limits)


def apply_cost(rid, cost, blocks, total_epsilon, slack, K):

    #max_cost = get_cost_limit(rid, blocks, total_epsilon, slack, K)
    #assert max_cost > cost or math.isclose(max_cost, cost), f"cannot allocate more than max cost  max={max_cost} vs. cost={cost}"

    for b in get_active_blocks(rid, blocks):
        assert rid not in b.allocations, "cannot allocate twice to same block in same round"
        b.allocations[rid] = cost


def spread_indices(length, n):
    if n <= 0 or length <= 0:
        return []

    step = length / n
    indices = [int(i * step) for i in range(n)]

    # Adjust last index to ensure it points to the last element
    indices[-1] = length - 1

    return indices


def get_colors(n):
    # '#2e222f', '#3e3546', '#625565',
    colors = [
        '#966c6c', '#ab947a', '#694f62', '#7f708a', '#9babb2', '#c7dcd0', '#ffffff',
        '#6e2727', '#b33831', '#ea4f36', '#f57d4a', '#ae2334', '#e83b3b', '#fb6b1d', '#f79617', '#f9c22b', '#7a3045',
        '#9e4539', '#cd683d', '#e6904e', '#fbb954', '#4c3e24', '#676633', '#a2a947', '#d5e04b', '#fbff86', '#165a4c',
        '#239063', '#1ebc73', '#91db69', '#cddf6c', '#313638', '#374e4a', '#547e64', '#92a984', '#b2ba90', '#0b5e65',
        '#0b8a8f', '#0eaf9b', '#30e1b9', '#8ff8e2', '#323353', '#484a77', '#4d65b4', '#4d9be6', '#8fd3ff', '#45293f',
        '#6b3e75', '#905ea9', '#a884f3', '#eaaded', '#753c54', '#a24b6f', '#cf657f', '#ed8099', '#831c5d', '#c32454',
        '#f04f78', '#f68181', '#fca790', '#fdcbb0'
    ]

    assert n <= len(colors), "not enough colors"

    step = len(colors) / n
    indices = [int(i * step) for i in range(n)]

    # Adjust last index to ensure it points to the last element
    indices[-1] = len(colors) - 1

    return [colors[i] for i in indices]