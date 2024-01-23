
import matplotlib.pyplot as plt

LABEL_EQUAL = '# requests'
LABEL_UTILITY = 'utility'


def get_labels():
    return {
        'dpf': 'DPF',
        'greedy': 'FCFS',
        'ilp': 'ILP',
        'ilp-equal': f'ILP ({LABEL_EQUAL})',
        'ilp-utility': f'ILP ({LABEL_UTILITY})',
        'ilp-equal-50': f'ILP50 ({LABEL_EQUAL})',
        'ilp-utility-50': f'ILP50 ({LABEL_UTILITY})',
        'ilp-equal-100': f'ILP100 ({LABEL_EQUAL})',
        'ilp-utility-100': f'ILP100 ({LABEL_UTILITY})',
        'ilp-equal-400': f'ILP400 ({LABEL_EQUAL})',
        'ilp-utility-400': f'ILP400 ({LABEL_UTILITY})',
        'weighted-dpf': 'DPF (W)',
        'dpk-gurobi': 'DPK (Gurobi)',
        'dpk-fptas': 'DPK (FPTAS)'
    }


def get_colors_bounds():
    color_map = {
        'dpf': '#C7B786',
        'greedy': '#D5E1A3',
        'ilp': (76 / 255.0, 114 / 255.0, 176 / 255.0),
        'ilp-equal': (166 / 255.0, 184 / 255.0, 216 / 255.0), # light blue no opacity
        'ilp-utility': (76 / 255.0, 114 / 255.0, 176 / 255.0),
        'ilp-equal-50': (166 / 255.0, 184 / 255.0, 216 / 255.0), # light blue no opacity
        'ilp-utility-50': (76 / 255.0, 114 / 255.0, 176 / 255.0),
        'ilp-equal-100': (166 / 255.0, 184 / 255.0, 216 / 255.0), # light blue no opacity
        'ilp-utility-100': (76 / 255.0, 114 / 255.0, 176 / 255.0),
        'ilp-equal-400': (166 / 255.0, 184 / 255.0, 216 / 255.0), # light blue no opacity
        'ilp-utility-400': (76 / 255.0, 114 / 255.0, 176 / 255.0),
        'fairness_pie': '#CCCCCC',
        "weighted-dpf": "#5dfc00",
        "dpk-gurobi": "#5dfcf7",
        "dpk-fptas": "#fd9ef7"
    }

    return color_map


def setup_plt():
    fig_size = [5.041760066417601, 3.8838667729342697]
    plt_params = {
        'backend': 'ps',
        'axes.labelsize': 18,
        'legend.fontsize': 12,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'font.size': 12,
        'figure.figsize': fig_size,
        'font.family': 'Times New Roman',
        'lines.markersize': 8
    }

    plt.rcParams.update(plt_params)
    plt.rc('pdf', fonttype=42)  # IMPORTANT to get rid of Type 3

def setup_plt_4by2():
    fig_size = [5.041760066417601, 3.8838667729342697]
    plt_params = {
        'backend': 'ps',
        'axes.labelsize': 18,
        'legend.fontsize': 12,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'font.size': 12,
        'figure.figsize': fig_size,
        'font.family': 'Times New Roman',
        'lines.markersize': 8
    }

    plt.rcParams.update(plt_params)
    plt.rc('pdf', fonttype=42)  # IMPORTANT to get rid of Type 3

def setup_plt_halfcolumn():
    fig_size = [3.441760066417601, 2.38667729342697]
    plt_params = {
        'backend': 'ps',
        'axes.labelsize': 18,
        'legend.fontsize': 12,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'font.size': 12,
        'figure.figsize': fig_size,
        'font.family': 'Times New Roman',
        'lines.markersize': 8
    }

    plt.rcParams.update(plt_params)
    plt.rc('pdf', fonttype=42)  # IMPORTANT to get rid of Type 3


def plt_adjust_edges(fig):
    fig.subplots_adjust(left=0.18, right=1.0, top=1.0, bottom=0.1)
