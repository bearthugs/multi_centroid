import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_gmm_quality(csv_path="cohesion_results.csv", output_dir="gmm_graphs"):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    # Clean up any bad rows
    df = df.dropna(subset=["avg_cohesion", "avg_separation", "ratio"])
    df = df[df["avg_cohesion"] != 0]

    # Derived column
    df["n_dim"] = df["n"] * df["dim"]

    # --- Shared style ---
    sns.set(style="whitegrid")
    plt.rcParams.update({"figure.figsize": (12, 8), "font.size": 13})

    # ---------- 1. K vs Cohesion ----------
    plt.figure()
    for dataset, group in df.groupby("dataset"):
        plt.plot(group["K"], group["avg_cohesion"], marker="o", label=dataset)
    plt.title("K vs Average Cohesion (within-cluster compactness)", fontsize=16)
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Average Cohesion")
    plt.yscale("log")
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "k_vs_cohesion.png"), dpi=300)
    plt.close()

    # ---------- 2. K vs Separation ----------
    plt.figure()
    for dataset, group in df.groupby("dataset"):
        plt.plot(group["K"], group["avg_separation"], marker="o", label=dataset)
    plt.title("K vs Average Separation (between-cluster distance)", fontsize=16)
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Average Separation")
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "k_vs_separation.png"), dpi=300)
    plt.close()

    # ---------- 3. K vs Ratio ----------
    plt.figure()
    for dataset, group in df.groupby("dataset"):
        plt.plot(group["K"], group["ratio"], marker="o", label=dataset)
    plt.title("K vs Separation/Cohesion Ratio", fontsize=16)
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Separation / Cohesion Ratio (higher = better)")
    plt.yscale("log")
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "k_vs_ratio.png"), dpi=300)
    plt.close()

    # ---------- 4. Cohesion vs Separation ----------
    plt.figure()
    sns.scatterplot(data=df, x="avg_cohesion", y="avg_separation", hue="dataset", style="K", s=100)
    plt.title("Cohesion vs Separation", fontsize=16)
    plt.xlabel("Average Cohesion (lower = tighter)")
    plt.ylabel("Average Separation (higher = better)")
    plt.xscale("log")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cohesion_vs_separation.png"), dpi=300)
    plt.close()

    # ---------- 5. n×dim vs Ratio ----------
    plt.figure()
    sns.lineplot(data=df, x="n_dim", y="ratio", hue="dataset", marker="o")
    plt.title("n×dim vs Separation/Cohesion Ratio", fontsize=16)
    plt.xlabel("n × dim (Dataset Scale × Dimensionality)")
    plt.ylabel("Separation / Cohesion Ratio")
    plt.xscale("log")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ndim_vs_ratio.png"), dpi=300)
    plt.close()

    print(f"Saved all GMM quality plots to: {output_dir}")

if __name__ == "__main__":
    plot_gmm_quality("cohesion_results.csv", "gmm_graphs")
