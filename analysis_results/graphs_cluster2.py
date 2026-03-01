import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_k_vs_query_time_dataset_tedge(csv_path="gmm_query.csv", output_dir="gmm_graphs"):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    # ------------------------------------------------------------
    # GLOBAL STYLE (consistent with your other plots)
    # ------------------------------------------------------------
    sns.set(style="whitegrid")
    plt.rcParams.update({
        "figure.figsize": (16, 10),
        "font.size": 18,
        "axes.titlesize": 26,
        "axes.titlepad": 14,
        "axes.labelsize": 20,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 13,
        "legend.title_fontsize": 16,
        "figure.constrained_layout.use": True,
    })

    # ------------------------------------------------------------
    # Style maps
    # ------------------------------------------------------------
    datasets = sorted(df["dataset"].unique())
    t_edges = sorted(df["t_edge"].unique())

    # Unique color per dataset
    palette = sns.color_palette("tab10", n_colors=len(datasets))
    dataset_colors = {ds: palette[i] for i, ds in enumerate(datasets)}

    # Markers + linestyles for t_edge
    markers = ['o', 's', '^', 'D', 'P']
    linestyles = ['-', '--', '-.', ':']
    t_edge_styles = {}
    for i, t in enumerate(t_edges):
        t_edge_styles[t] = (markers[i % len(markers)],
                            linestyles[i % len(linestyles)])

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------
    fig, ax = plt.subplots(layout="constrained")

    for ds in datasets:
        ds_group = df[df["dataset"] == ds]
        color = dataset_colors[ds]

        for t in t_edges:
            g = ds_group[ds_group["t_edge"] == t].sort_values("K")
            if len(g) == 0:
                continue
            
            marker, linestyle = t_edge_styles[t]

            ax.plot(
                g["K"],
                g["query_time_ms"],
                color=color,
                marker=marker,
                linestyle=linestyle,
                markersize=7,
                linewidth=2.0,
                label=f"{ds}, t_edge={t}"
            )

    ax.set_title("K vs Query Time (grouped by dataset and t_edge)")
    ax.set_xlabel("Number of Clusters (K)")
    ax.set_ylabel("Query Time (ms)")
    ax.set_yscale("log")

    # Big legend
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.25),
        ncol=3,
        frameon=False
    )

    fig.savefig(
        os.path.join(output_dir, "global_K_vs_query_time_by_dataset_tedge.png"),
        dpi=300, bbox_inches="tight"
    )
    plt.close()

    print(f"Created global dataset+t_edge grouped plot in: {output_dir}")


def plot_k_vs_metrics_per_dataset(csv_path="gmm_query.csv", output_dir="gmm_graphs"):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    # ------------------------------------------------------------
    # GLOBAL STYLE (same as GMM graphs)
    # ------------------------------------------------------------
    sns.set(style="whitegrid")
    plt.rcParams.update({
        "figure.figsize": (14, 9),
        "font.size": 18,
        "axes.titlesize": 26,
        "axes.titlepad": 14,
        "axes.labelsize": 20,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 15,
        "legend.title_fontsize": 18,
        "figure.constrained_layout.use": True,
    })

    # ------------------------------------------------------------
    # Style maps
    # ------------------------------------------------------------
    t_edges = sorted(df["t_edge"].unique())

    # Colors = t_edge
    palette = sns.color_palette(n_colors=len(t_edges))
    t_edge_colors = {t: palette[i] for i, t in enumerate(t_edges)}

    # Markers + linestyles per t_edge
    markers = ['o', 's', '^']
    linestyles = ['-', '--', '-.']
    t_edge_styles = {}
    for i, t in enumerate(t_edges):
        t_edge_styles[t] = (markers[i % len(markers)],
                            linestyles[i % len(linestyles)])

    # Legend helper
    def bottom_legend(ax):
        ax.legend(
            loc='upper center',
            bbox_to_anchor=(0.5, -0.20),
            ncol=3,
            frameon=False
        )

    # ============================================================
    # PER-DATASET PLOTS
    # ============================================================
    for dataset, ds_group in df.groupby("dataset"):

        # --------------------------------------------------------
        # 1. K vs QUERY TIME
        # --------------------------------------------------------
        fig, ax = plt.subplots(layout="constrained")

        for t in t_edges:
            g = ds_group[ds_group["t_edge"] == t].sort_values("K")

            if len(g) == 0:
                continue

            color = t_edge_colors[t]
            marker, linestyle = t_edge_styles[t]

            ax.plot(
                g["K"],
                g["query_time_ms"],
                color=color,
                marker=marker,
                linestyle=linestyle,
                markersize=8,
                linewidth=2.2,
                label=f"t_edge={t}"
            )

        ax.set_title(f"{dataset}: K vs Query Time")
        ax.set_xlabel("Number of Clusters (K)")
        ax.set_ylabel("Query Time (ms)")
        ax.set_yscale("log")

        bottom_legend(ax)

        fig.savefig(
            os.path.join(output_dir, f"{dataset}_K_vs_query_time.png"),
            dpi=300, bbox_inches="tight"
        )
        plt.close()

        # --------------------------------------------------------
        # 2. K vs RECALL
        # --------------------------------------------------------
        fig, ax = plt.subplots(layout="constrained")

        for t in t_edges:
            g = ds_group[ds_group["t_edge"] == t].sort_values("K")

            if len(g) == 0:
                continue

            color = t_edge_colors[t]
            marker, linestyle = t_edge_styles[t]

            ax.plot(
                g["K"],
                g["accuracy"],
                color=color,
                marker=marker,
                linestyle=linestyle,
                markersize=8,
                linewidth=2.2,
                label=f"t_edge={t}"
            )

        ax.set_title(f"{dataset}: K vs Recall")
        ax.set_xlabel("Number of Clusters (K)")
        ax.set_ylabel("Recall")

        bottom_legend(ax)

        fig.savefig(
            os.path.join(output_dir, f"{dataset}_K_vs_recall.png"),
            dpi=300, bbox_inches="tight"
        )
        plt.close()

    print(f"Created per-dataset K vs metrics plots in: {output_dir}")


if __name__ == "__main__":
    plot_k_vs_metrics_per_dataset()
    plot_k_vs_query_time_dataset_tedge()
