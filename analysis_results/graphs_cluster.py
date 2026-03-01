import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_gmm_quality(csv_path="cohesion_results.csv", output_dir="gmm_graphs"):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    # Clean up bad rows
    df = df.dropna(subset=["avg_cohesion", "avg_separation", "ratio"])
    df = df[df["avg_cohesion"] != 0]

    # Derived column
    df["n_dim"] = df["n"] * df["dim"]

    # ------------------------------------------------------------
    # GLOBAL STYLE CONFIGURATION
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
    # DISTINCT MARKERS + LINESTYLES
    # ------------------------------------------------------------
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'P', 'X', '*', 'H']
    linestyles = ['-', '--', '-.', ':']

    datasets = list(df['dataset'].unique())
    style_map = {}
    for i, dataset in enumerate(datasets):
        marker = markers[i % len(markers)]
        linestyle = linestyles[i % len(linestyles)]
        style_map[dataset] = (marker, linestyle)

    # ------------------------------------------------------------
    # HELPER TO ADD LEGEND AT BOTTOM
    # ------------------------------------------------------------
    def bottom_legend():
        plt.legend(
            loc='upper center',
            bbox_to_anchor=(0.5, -0.20),  # move down further
            ncol=3,
            frameon=False
        )

    # ------------------------------------------------------------
    # HELPER: LINE PLOTS WITH BOTTOM LEGEND
    # ------------------------------------------------------------
    def plot_lines(y_col, ylabel, title, filename, logy=True):
        fig, ax = plt.subplots(layout="constrained")

        for dataset, group in df.groupby("dataset"):
            group = group.sort_values("K")
            marker, linestyle = style_map[dataset]

            ax.plot(
                group["K"], group[y_col],
                marker=marker,
                linestyle=linestyle,
                markersize=8,
                linewidth=2.2,
                label=dataset
            )

        ax.set_title(title)
        ax.set_xlabel("Number of Clusters (K)")
        ax.set_ylabel(ylabel)

        if logy:
            ax.set_yscale("log")

        bottom_legend()

        fig.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches="tight")
        plt.close()

    # ------------------------------------------------------------
    # 1. K vs Cohesion
    # ------------------------------------------------------------
    plot_lines(
        y_col="avg_cohesion",
        ylabel="Average Cohesion",
        title="K vs Average Cohesion (within-cluster compactness)",
        filename="k_vs_cohesion.png"
    )

    # ------------------------------------------------------------
    # 2. K vs Separation
    # ------------------------------------------------------------
    plot_lines(
        y_col="avg_separation",
        ylabel="Average Separation",
        title="K vs Average Separation (between-cluster distance)",
        filename="k_vs_separation.png"
    )

    # ------------------------------------------------------------
    # 3. K vs Ratio
    # ------------------------------------------------------------
    plot_lines(
        y_col="ratio",
        ylabel="Separation/Cohesion Ratio (higher = better)",
        title="K vs Separation/Cohesion Ratio",
        filename="k_vs_ratio.png"
    )

    # ------------------------------------------------------------
    # 4. Cohesion vs Separation (scatter plot)
    # ------------------------------------------------------------
    fig, ax = plt.subplots(layout="constrained")

    sns.scatterplot(
        data=df,
        x="avg_cohesion",
        y="avg_separation",
        hue="dataset",
        style="K",
        s=140,
        ax=ax
    )

    ax.set_title("Cohesion vs Separation")
    ax.set_xlabel("Average Cohesion (lower = tighter)")
    ax.set_ylabel("Average Separation (higher = better)")
    ax.set_xscale("log")

    # Move legend bottom
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.20),
        ncol=3,
        frameon=False
    )

    fig.savefig(os.path.join(output_dir, "cohesion_vs_separation.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # ------------------------------------------------------------
    # 5. n × dim vs Ratio
    # ------------------------------------------------------------
    fig, ax = plt.subplots(layout="constrained")

    for dataset, group in df.groupby("dataset"):
        group = group.sort_values("n_dim")
        marker, linestyle = style_map[dataset]

        ax.plot(
            group["n_dim"],
            group["ratio"],
            marker=marker,
            linestyle=linestyle,
            markersize=8,
            linewidth=2.2,
            label=dataset
        )

    ax.set_title("n × dim vs Separation/Cohesion Ratio")
    ax.set_xlabel("n × dim (Dataset Scale × Dimensionality)")
    ax.set_ylabel("Separation/Cohesion Ratio")
    ax.set_xscale("log")
    ax.set_yscale("log")

    bottom_legend()

    fig.savefig(os.path.join(output_dir, "ndim_vs_ratio.png"), dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved all GMM quality plots to: {output_dir}")


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
if __name__ == "__main__":
    plot_gmm_quality("cohesion_results.csv", "gmm_graphs")
