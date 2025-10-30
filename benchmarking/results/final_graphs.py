import pandas as pd
import matplotlib.pyplot as plt
import os


def plot_construction_time(csv_path='results.csv', ef_values=[50, 100, 200], M_values=None):
    """Plot M vs Construction Time for each efConstruction value (fixed y-axis scale)."""
    output_dir = "final_graphs"
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    if M_values is not None:
        df = df[df['M'].isin(M_values)]
    if ef_values is not None:
        df = df[df['efConstruction'].isin(ef_values)]

    y_min, y_max = 1950, 9e7  # Fixed range for consistency

    for ef in sorted(df['efConstruction'].unique()):
        subset = df[df['efConstruction'] == ef]
        plt.figure(figsize=(12, 8))
        for dataset, group in subset.groupby('dataset'):
            group = group.sort_values('M')
            plt.plot(group['M'], group['construction_time(ms)'], marker='o', label=dataset)
        
        plt.title(f"M vs Construction Time (efConstruction = {ef})", fontsize=16)
        plt.xlabel("M", fontsize=14)
        plt.ylabel("Construction Time (ms)", fontsize=14)
        plt.yscale("log")
        plt.ylim(y_min, y_max)
        plt.legend(ncol=2, fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

        filename = f"{output_dir}/M_vs_construction_time_ef{ef}.png"
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"Saved: {filename}")


def plot_query_time(csv_path='results.csv', ef_values=[50, 100, 200], M_values=None):
    """Plot M vs Query Time for each efConstruction value (fixed y-axis scale)."""
    output_dir = "final_graphs"
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    if M_values is not None:
        df = df[df['M'].isin(M_values)]
    if ef_values is not None:
        df = df[df['efConstruction'].isin(ef_values)]

    y_min, y_max = 0, 10000  # Fixed range for comparability

    for ef in sorted(df['efConstruction'].unique()):
        subset = df[df['efConstruction'] == ef]
        plt.figure(figsize=(12, 8))
        for dataset, group in subset.groupby('dataset'):
            group = group.sort_values('M')
            plt.plot(group['M'], group['query_time(ms)'], marker='o', label=dataset)
        
        plt.title(f"M vs Query Time (efConstruction = {ef})", fontsize=16)
        plt.xlabel("M", fontsize=14)
        plt.ylabel("Query Time (ms)", fontsize=14)
        plt.ylim(y_min, y_max)
        plt.legend(ncol=2, fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

        filename = f"{output_dir}/M_vs_query_time_ef{ef}.png"
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"Saved: {filename}")


def plot_recall(csv_path='results.csv', ef_values=[50, 100, 200], M_values=None):
    """Plot M vs Recall@100 for each efConstruction value (fixed y-axis scale)."""
    output_dir = "final_graphs"
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    if M_values is not None:
        df = df[df['M'].isin(M_values)]
    if ef_values is not None:
        df = df[df['efConstruction'].isin(ef_values)]

    y_min, y_max = 0, 1.01  # Recall always between 0–1

    for ef in sorted(df['efConstruction'].unique()):
        subset = df[df['efConstruction'] == ef]
        plt.figure(figsize=(12, 8))
        for dataset, group in subset.groupby('dataset'):
            group = group.sort_values('M')
            plt.plot(group['M'], group['recall@100'], marker='o', label=dataset)
        
        plt.title(f"M vs Recall@100 (efConstruction = {ef})", fontsize=16)
        plt.xlabel("M", fontsize=14)
        plt.ylabel("Recall@100", fontsize=14)
        plt.ylim(y_min, y_max)
        plt.legend(ncol=2, fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

        filename = f"{output_dir}/M_vs_recall_ef{ef}.png"
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"Saved: {filename}")


def plot_construction_time_together(csv_path='results.csv', ef_values=[50, 100, 200], M_values=None):
    """Plot M vs Construction Time across multiple efConstruction values (legend at bottom with spacing)."""
    output_dir = "final_graphs"
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    if M_values is not None:
        df = df[df['M'].isin(M_values)]

    y_min, y_max = 1950, 9e7  # Fixed y-axis range for comparability

    fig, axes = plt.subplots(1, len(ef_values), figsize=(18, 6), sharey=True)
    for i, ef in enumerate(ef_values):
        ax = axes[i]
        subset = df[df['efConstruction'] == ef]
        for dataset, group in subset.groupby('dataset'):
            group = group.sort_values('M')
            ax.plot(group['M'], group['construction_time(ms)'], marker='o', label=dataset)
        
        ax.set_title(f"efConstruction = {ef}", fontsize=14)
        ax.set_xlabel("M", fontsize=12)
        plt.yscale("log")
        ax.set_ylim(y_min, y_max)
        ax.grid(True, linestyle='--', alpha=0.7)
        if i == 0:
            ax.set_ylabel("Construction Time (ms)", fontsize=12)
    
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=9, frameon=False)
    fig.suptitle("M vs Construction Time", fontsize=16)
    plt.tight_layout(rect=[0, 0.12, 1, 0.93])  # increased bottom margin
    filename = f"{output_dir}/M_vs_construction_time_all_ef.png"
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved: {filename}")


def plot_query_time_together(csv_path='results.csv', ef_values=[50, 100, 200], M_values=None):
    """Plot M vs Query Time across multiple efConstruction values (legend at bottom with spacing)."""
    output_dir = "final_graphs"
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    if M_values is not None:
        df = df[df['M'].isin(M_values)]

    y_min, y_max = 0, 10000  # Fixed range for comparability

    fig, axes = plt.subplots(1, len(ef_values), figsize=(18, 6), sharey=True)
    for i, ef in enumerate(ef_values):
        ax = axes[i]
        subset = df[df['efConstruction'] == ef]
        for dataset, group in subset.groupby('dataset'):
            group = group.sort_values('M')
            ax.plot(group['M'], group['query_time(ms)'], marker='o', label=dataset)
        
        ax.set_title(f"efConstruction = {ef}", fontsize=14)
        ax.set_xlabel("M", fontsize=12)
        ax.set_ylim(y_min, y_max)
        ax.grid(True, linestyle='--', alpha=0.7)
        if i == 0:
            ax.set_ylabel("Query Time (ms)", fontsize=12)
    
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=9, frameon=False)
    fig.suptitle("M vs Query Time", fontsize=16)
    plt.tight_layout(rect=[0, 0.12, 1, 0.93])  # increased bottom margin
    filename = f"{output_dir}/M_vs_query_time_all_ef.png"
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved: {filename}")


def plot_recall_together(csv_path='results.csv', ef_values=[50, 100, 200], M_values=None):
    """Plot M vs Recall@100 across multiple efConstruction values (legend at bottom with spacing)."""
    output_dir = "final_graphs"
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    if M_values is not None:
        df = df[df['M'].isin(M_values)]

    y_min, y_max = 0.5, 1.01  # Fixed recall range

    fig, axes = plt.subplots(1, len(ef_values), figsize=(18, 6), sharey=True)
    for i, ef in enumerate(ef_values):
        ax = axes[i]
        subset = df[df['efConstruction'] == ef]
        for dataset, group in subset.groupby('dataset'):
            group = group.sort_values('M')
            ax.plot(group['M'], group['recall@100'], marker='o', label=dataset)
        
        ax.set_title(f"efConstruction = {ef}", fontsize=14)
        ax.set_xlabel("M", fontsize=12)
        ax.set_ylim(y_min, y_max)
        ax.grid(True, linestyle='--', alpha=0.7)
        if i == 0:
            ax.set_ylabel("Recall@100", fontsize=12)
    
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=9, frameon=False)
    fig.suptitle("M vs Recall@100", fontsize=16)
    plt.tight_layout(rect=[0, 0.12, 1, 0.93])  # increased bottom margin
    filename = f"{output_dir}/M_vs_recall_all_ef.png"
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved: {filename}")


# --- Example usage --- #
plot_construction_time('results.csv')
plot_query_time('results.csv')
plot_recall('results.csv')
plot_construction_time_together('results.csv')
plot_query_time_together('results.csv')
plot_recall_together('results.csv')
