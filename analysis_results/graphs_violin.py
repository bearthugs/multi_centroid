import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_cluster_distributions(index_dir, K, output_dir="gmm_graphs"):
    """
    Parse cluster_to_vectors index files and plot violin plots showing
    cluster size distributions for all datasets.
    """
    os.makedirs(output_dir, exist_ok=True)

    all_data = []

    # Iterate through all index files
    for fname in os.listdir(index_dir):
        if not fname.endswith(".index"):
            continue

        dataset_name = fname.replace(".index", "")
        path = os.path.join(index_dir, fname)

        cluster_sizes = []
        with open(path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) > 1:
                    cluster_sizes.append(len(parts) - 1)  # subtract cluster ID itself
                else:
                    cluster_sizes.append(0)

        for size in cluster_sizes:
            all_data.append({"dataset": dataset_name, "cluster_size": size})

    if not all_data:
        print("No cluster_to_vectors index files found.")
        return

    df = pd.DataFrame(all_data)

    # === Plot Violin Plot ===
    plt.figure(figsize=(14, 8))
    sns.violinplot(
        data=df,
        x="dataset",
        y="cluster_size",
        inner="box",
        density_norm="width",
        bw_adjust=1.5,
        cut=1,
        gridsize=200,
    )
    plt.ylim(bottom=0)
    plt.title(f"Distribution of Cluster Sizes Across Datasets ({K})", fontsize=16)
    plt.xlabel("Dataset", fontsize=14)
    plt.ylabel("Cluster Size (# of assigned vectors)", fontsize=14)
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()

    output_path = os.path.join(output_dir, f"cluster_violin_{K}.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved violin plot: {output_path}")


if __name__ == "__main__":
    plot_cluster_distributions("../gmm_indexes/cluster_to_vectors/K8", "K8")
    plot_cluster_distributions("../gmm_indexes/cluster_to_vectors/K16", "K16")
    plot_cluster_distributions("../gmm_indexes/cluster_to_vectors/K32", "K32")
    plot_cluster_distributions("../gmm_indexes/cluster_to_vectors/K64", "K64")
    plot_cluster_distributions("../gmm_indexes/cluster_to_vectors/K128", "K128")
