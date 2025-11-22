import pandas as pd
import matplotlib.pyplot as plt
import os

# -------------------------------
# GLOBAL STYLE IMPROVEMENTS
# -------------------------------

plt.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "legend.fontsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14
})

MARKERS = ['o', 's', '^', 'D', 'P', '*', 'X', 'v', '<', '>', 'h']

def marker_for_dataset(name):
    """Assign a consistent marker shape based on dataset name."""
    idx = abs(hash(name)) % len(MARKERS)
    return MARKERS[idx]


# -------------------------------
# INDIVIDUAL PLOTS
# -------------------------------

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
            plt.plot(
                group['M'],
                group['construction_time(ms)'],
                marker=marker_for_dataset(dataset),
                linestyle=':',
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
            plt.plot(
                group['M'],
                group['query_time(ms)'],
                marker=marker_for_dataset(dataset),
                linestyle=':',
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
            plt.plot(
                group['M'],
                group['recall@100'],
                marker=marker_for_dataset(dataset),
                linestyle=':',
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


# -------------------------------
# COMBINED PLOTS (multi-subplot)
# -------------------------------

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
            ax.plot(
                group['M'],
                group['construction_time(ms)'],
                marker=marker_for_dataset(dataset),
                linestyle=':',
                linewidth=2,
                markersize=8,
                label=dataset
            )
        
        ax.set_title(f"efConstruction = {ef}")
        ax.set_xlabel("M")
        ax.set_yscale("log")
        ax.set_ylim(y_min, y_max)
        ax.grid(True, linestyle='--', alpha=0.7)
        if i == 0:
            ax.set_ylabel("Construction Time (ms)")
    
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, frameon=False)
    fig.suptitle("M vs Construction Time")
    plt.tight_layout(rect=[0, 0.12, 1, 0.93])
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
            ax.plot(
                group['M'],
                group['query_time(ms)'],
                marker=marker_for_dataset(dataset),
                linestyle=':',
                linewidth=2,
                markersize=8,
                label=dataset
            )
        
        ax.set_title(f"efConstruction = {ef}")
        ax.set_xlabel("M")
        ax.set_ylim(y_min, y_max)
        ax.grid(True, linestyle='--', alpha=0.7)
        if i == 0:
            ax.set_ylabel("Query Time (ms)")
    
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, frameon=False)
    fig.suptitle("M vs Query Time")
    plt.tight_layout(rect=[0, 0.12, 1, 0.93])
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
            ax.plot(
                group['M'],
                group['recall@100'],
                marker=marker_for_dataset(dataset),
                linestyle=':',
                linewidth=2,
                markersize=8,
                label=dataset
            )
        
        ax.set_title(f"efConstruction = {ef}")
        ax.set_xlabel("M")
        ax.set_ylim(y_min, y_max)
        ax.grid(True, linestyle='--', alpha=0.7)
        if i == 0:
            ax.set_ylabel("Recall")
    
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, frameon=False)
    fig.suptitle("M vs Recall (efSearch = 200)")
    plt.tight_layout(rect=[0, 0.12, 1, 0.93])
    filename = f"{output_dir}/M_vs_recall_all_ef.png"
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved: {filename}")


# -------------------------------
# AUTO-RUN (same as original)
# -------------------------------

plot_construction_time('results.csv')
plot_query_time('results.csv')
plot_recall('results.csv')
plot_construction_time_together('results.csv')
plot_query_time_together('results.csv')
plot_recall_together('results.csv')
