import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_time_accuracy_global(csv_path="gmm_query.csv", output_dir="gmm_graphs"):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    # ------------------------------------------------------------
    # GLOBAL STYLE (MATCHES GMM PLOTS)
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
    # COLOR = DATASET
    # ------------------------------------------------------------
    datasets = sorted(df["dataset"].unique())
    palette = sns.color_palette(n_colors=len(datasets))
    color_map = {ds: palette[i] for i, ds in enumerate(datasets)}

    # ------------------------------------------------------------
    # MARKER + LINESTYLE = K
    # ------------------------------------------------------------
    Ks = sorted(df["K"].unique())
    markers = ['o', 's', '^', 'D', 'P', 'X']
    linestyles = ['-', '--', '-.', ':']

    style_map = {}
    for i, K in enumerate(Ks):
        marker = markers[i % len(markers)]
        linestyle = linestyles[i % len(linestyles)]
        style_map[K] = (marker, linestyle)

    # bottom legend helper
    def bottom_legend(ax):
        ax.legend(
            loc='upper center',
            bbox_to_anchor=(0.5, -0.20),
            ncol=3,
            frameon=False
        )

    # ============================================================
    # A. GLOBAL PLOT — t_edge vs Accuracy
    # ============================================================
    fig, ax = plt.subplots(layout="constrained")

    for dataset in datasets:
        ds_group = df[df["dataset"] == dataset]

        for K in Ks:
            group = ds_group[ds_group["K"] == K].sort_values("t_edge")

            if len(group) == 0:
                continue

            color = color_map[dataset]
            marker, linestyle = style_map[K]

            ax.plot(
                group["t_edge"],
                group["accuracy"],
                label=f"{dataset}, K={K}",
                color=color,
                marker=marker,
                linestyle=linestyle,
                markersize=8,
                linewidth=2.2,
            )

    ax.set_title("t_edge vs Recall (All Datasets)")
    ax.set_xlabel("t_edge Threshold")
    ax.set_ylabel("Recall")
    bottom_legend(ax)

    fig.savefig(
        os.path.join(output_dir, "tEdge_vs_recall_ALL.png"),
        dpi=300, bbox_inches="tight"
    )
    plt.close()

    # ============================================================
    # B. GLOBAL PLOT — t_edge vs Query Time
    # ============================================================
    fig, ax = plt.subplots(layout="constrained")

    for dataset in datasets:
        ds_group = df[df["dataset"] == dataset]

        for K in Ks:
            group = ds_group[ds_group["K"] == K].sort_values("t_edge")

            if len(group) == 0:
                continue

            color = color_map[dataset]
            marker, linestyle = style_map[K]

            ax.plot(
                group["t_edge"],
                group["query_time_ms"],
                label=f"{dataset}, K={K}",
                color=color,
                marker=marker,
                linestyle=linestyle,
                markersize=8,
                linewidth=2.2,
            )

    ax.set_title("t_edge vs Query Time (All Datasets)")
    ax.set_xlabel("t_edge Threshold")
    ax.set_ylabel("Query Time (ms)")
    ax.set_yscale("log")
    bottom_legend(ax)

    fig.savefig(
        os.path.join(output_dir, "tEdge_vs_query_time_ALL.png"),
        dpi=300, bbox_inches="tight"
    )
    plt.close()

    # ============================================================
    # C. PER-DATASET PLOTS — NEW SECTION
    # ============================================================
    for dataset in datasets:
        ds_group = df[df["dataset"] == dataset]

        # ------------------ Recall ------------------
        fig, ax = plt.subplots(layout="constrained")

        for K in Ks:
            g = ds_group[ds_group["K"] == K].sort_values("t_edge")
            if len(g) == 0:
                continue

            marker, linestyle = style_map[K]

            ax.plot(
                g["t_edge"],
                g["accuracy"],
                marker=marker,
                linestyle=linestyle,
                markersize=8,
                linewidth=2.2,
                label=f"K={K}"
            )

        ax.set_title(f"{dataset}: t_edge vs Recall")
        ax.set_xlabel("t_edge Threshold")
        ax.set_ylabel("Recall")
        bottom_legend(ax)

        fig.savefig(
            os.path.join(output_dir, f"{dataset}_tEdge_vs_recall.png"),
            dpi=300, bbox_inches="tight"
        )
        plt.close()

        # ------------------ Query Time ------------------
        fig, ax = plt.subplots(layout="constrained")

        for K in Ks:
            g = ds_group[ds_group["K"] == K].sort_values("t_edge")
            if len(g) == 0:
                continue

            marker, linestyle = style_map[K]

            ax.plot(
                g["t_edge"],
                g["query_time_ms"],
                marker=marker,
                linestyle=linestyle,
                markersize=8,
                linewidth=2.2,
                label=f"K={K}"
            )

        ax.set_title(f"{dataset}: t_edge vs Query Time")
        ax.set_xlabel("t_edge Threshold")
        ax.set_ylabel("Query Time (ms)")
        ax.set_yscale("log")
        bottom_legend(ax)

        fig.savefig(
            os.path.join(output_dir, f"{dataset}_tEdge_vs_query_time.png"),
            dpi=300, bbox_inches="tight"
        )
        plt.close()

    print(f"Saved all global and per-dataset plots to: {output_dir}")


if __name__ == "__main__":
    plot_time_accuracy_global("gmm_query.csv", "gmm_graphs")
