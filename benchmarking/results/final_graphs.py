import pandas as pd
import matplotlib.pyplot as plt
import os

# -------------------------------------------------------
# GLOBAL STYLE IMPROVEMENTS
# -------------------------------------------------------

plt.rcParams.update({
    "font.size": 20,
    "axes.titlesize": 22,
    "axes.labelsize": 20,
    "legend.fontsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16
})

MARKERS = ['o', 's', '^', 'D', 'P', '*', 'X', 'v', '<', '>', 'h']
LINESTYLES = ['-', '--', '-.', ':']


def style_for_dataset(name):
    """Assign a consistent marker + linestyle based on dataset name."""
    h = abs(hash(name))
    marker = MARKERS[h % len(MARKERS)]
    linestyle = LINESTYLES[(h // len(MARKERS)) % len(LINESTYLES)]
    return marker, linestyle


# -------------------------------------------------------
# INDIVIDUAL PLOTS
# -------------------------------------------------------

def plot_construction_time(csv_path='results.csv', ef_values=[50, 100, 200], M_values=None):
    output_dir = "final_graphs"
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    if M_values is not None:
        df = df[df['M'].isin(M_values)]
    if ef_values is not None:
        df = df[df['efConstruction'].isin(ef_values)]

    y_min, y_max = 1950, 9e7

    for ef in sorted(df['efConstruction'].unique()):
        subset = df[df['efConstruction'] == ef]
        plt.figure(figsize=(12, 8))

        for dataset, group in subset.groupby('dataset'):
            group = group.sort_values('M')
            marker, linestyle = style_for_dataset(dataset)

            plt.plot(
                group['M'],
                group['construction_time(ms)'],
                marker=marker,
                linestyle=linestyle,
                linewidth=2,
                markersize=8,
                label=dataset
            )

        plt.title(f"M vs Construction Time (efConstruction = {ef})")
        plt.xlabel("M")
        plt.ylabel("Construction Time (ms)")
        plt.yscale("log")
        plt.ylim(y_min, y_max)
        plt.legend(ncol=2)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

        filename = f"{output_dir}/M_vs_construction_time_ef{ef}.png"
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"Saved: {filename}")


def plot_query_time(csv_path='results.csv', ef_values=[50, 100, 200], M_values=None):
    output_dir = "final_graphs"
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    if M_values is not None:
        df = df[df['M'].isin(M_values)]
    if ef_values is not None:
        df = df[df['efConstruction'].isin(ef_values)]

    y_min, y_max = 0, 10000

    for ef in sorted(df['efConstruction'].unique()):
        subset = df[df['efConstruction'] == ef]
        plt.figure(figsize=(12, 8))

        for dataset, group in subset.groupby('dataset'):
            group = group.sort_values('M')
            marker, linestyle = style_for_dataset(dataset)

            plt.plot(
                group['M'],
                group['query_time(ms)'],
                marker=marker,
                linestyle=linestyle,
                linewidth=2,
                markersize=8,
                label=dataset
            )

        plt.title(f"M vs Query Time (efConstruction = {ef})")
        plt.xlabel("M")
        plt.ylabel("Query Time (ms)")
        plt.ylim(y_min, y_max)
        plt.legend(ncol=2)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

        filename = f"{output_dir}/M_vs_query_time_ef{ef}.png"
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"Saved: {filename}")


def plot_recall(csv_path='results.csv', ef_values=[50, 100, 200], M_values=None):
    output_dir = "final_graphs"
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    if M_values is not None:
        df = df[df['M'].isin(M_values)]
    if ef_values is not None:
        df = df[df['efConstruction'].isin(ef_values)]

    y_min, y_max = 0.5, 1.01

    for ef in sorted(df['efConstruction'].unique()):
        subset = df[df['efConstruction'] == ef]
        plt.figure(figsize=(12, 8))

        for dataset, group in subset.groupby('dataset'):
            group = group.sort_values('M')
            marker, linestyle = style_for_dataset(dataset)

            plt.plot(
                group['M'],
                group['recall@100'],
                marker=marker,
                linestyle=linestyle,
                linewidth=2,
                markersize=8,
                label=dataset
            )

        plt.title(f"M vs Recall (efConstruction = {ef}, efSearch = 200)")
        plt.xlabel("M")
        plt.ylabel("Recall")
        plt.ylim(y_min, y_max)
        plt.legend(ncol=2)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

        filename = f"{output_dir}/M_vs_recall_ef{ef}.png"
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"Saved: {filename}")


# -------------------------------------------------------
# MULTI-SUBPLOT PLOTS
# -------------------------------------------------------

def plot_construction_time_together(csv_path='results.csv', ef_values=[50, 100, 200], M_values=None):
    output_dir = "final_graphs"
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    if M_values is not None:
        df = df[df['M'].isin(M_values)]

    y_min, y_max = 1950, 9e7

    fig, axes = plt.subplots(1, len(ef_values), figsize=(18, 6), sharey=True)

    for i, ef in enumerate(ef_values):
        ax = axes[i]
        subset = df[df['efConstruction'] == ef]

        for dataset, group in subset.groupby('dataset'):
            group = group.sort_values('M')
            marker, linestyle = style_for_dataset(dataset)

            ax.plot(
                group['M'],
                group['construction_time(ms)'],
                marker=marker,
                linestyle=linestyle,
                linewidth=2,
                markersize=8,
                label=dataset
            )

        ax.set_title(f"efConstruction = {ef}", pad=8)
        ax.set_xlabel("M")
        ax.set_yscale("log")
        ax.set_ylim(y_min, y_max)
        ax.grid(True, linestyle='--', alpha=0.7)
        if i == 0:
            ax.set_ylabel("Construction Time (ms)")

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, frameon=False)
    fig.suptitle("M vs Construction Time", y=0.97)
    plt.tight_layout(rect=[0, 0.20, 1, 0.92])

    filename = f"{output_dir}/M_vs_construction_time_all_ef.png"
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved: {filename}")


def plot_query_time_together(csv_path='results.csv', ef_values=[50, 100, 200], M_values=None):
    output_dir = "final_graphs"
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    if M_values is not None:
        df = df[df['M'].isin(M_values)]

    y_min, y_max = 0, 10000

    fig, axes = plt.subplots(1, len(ef_values), figsize=(18, 6), sharey=True)

    for i, ef in enumerate(ef_values):
        ax = axes[i]
        subset = df[df['efConstruction'] == ef]

        for dataset, group in subset.groupby('dataset'):
            group = group.sort_values('M')
            marker, linestyle = style_for_dataset(dataset)

            ax.plot(
                group['M'],
                group['query_time(ms)'],
                marker=marker,
                linestyle=linestyle,
                linewidth=2,
                markersize=8,
                label=dataset
            )

        ax.set_title(f"efConstruction = {ef}", pad=8)
        ax.set_xlabel("M")
        ax.set_ylim(y_min, y_max)
        ax.grid(True, linestyle='--', alpha=0.7)
        if i == 0:
            ax.set_ylabel("Query Time (ms)")

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, frameon=False)
    fig.suptitle("M vs Query Time", y=0.97)
    plt.tight_layout(rect=[0, 0.20, 1, 0.92])

    filename = f"{output_dir}/M_vs_query_time_all_ef.png"
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved: {filename}")


def plot_recall_together(csv_path='results.csv', ef_values=[50, 100, 200], M_values=None):
    output_dir = "final_graphs"
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    if M_values is not None:
        df = df[df['M'].isin(M_values)]

    y_min, y_max = 0.5, 1.01

    fig, axes = plt.subplots(1, len(ef_values), figsize=(18, 6), sharey=True)

    for i, ef in enumerate(ef_values):
        ax = axes[i]
        subset = df[df['efConstruction'] == ef]

        for dataset, group in subset.groupby('dataset'):
            group = group.sort_values('M')
            marker, linestyle = style_for_dataset(dataset)

            ax.plot(
                group['M'],
                group['recall@100'],
                marker=marker,
                linestyle=linestyle,
                linewidth=2,
                markersize=8,
                label=dataset
            )

        ax.set_title(f"efConstruction = {ef}", pad=8)
        ax.set_xlabel("M")
        ax.set_ylim(y_min, y_max)
        ax.grid(True, linestyle='--', alpha=0.7)
        if i == 0:
            ax.set_ylabel("Recall")

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, frameon=False)
    fig.suptitle("M vs Recall (efSearch = 200)", y=0.97)
    plt.tight_layout(rect=[0, 0.20, 1, 0.92])

    filename = f"{output_dir}/M_vs_recall_all_ef.png"
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved: {filename}")


# -------------------------------------------------------
# AUTO-RUN
# -------------------------------------------------------

plot_construction_time('results.csv')
plot_query_time('results.csv')
plot_recall('results.csv')
plot_construction_time_together('results.csv')
plot_query_time_together('results.csv')
plot_recall_together('results.csv')

