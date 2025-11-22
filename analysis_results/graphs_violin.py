import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_cluster_distributions(index_dir, K, output_dir="gmm_graphs"):
    os.makedirs(output_dir, exist_ok=True)

    all_data = []

    # --- Load data ---
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
                    cluster_sizes.append(len(parts) - 1)
                else:
                    cluster_sizes.append(0)

        for size in cluster_sizes:
            all_data.append({"dataset": dataset_name, "cluster_size": size})

    if not all_data:
        print("No cluster_to_vectors index files found.")
        return

    df = pd.DataFrame(all_data)

    # -----------------------------------------------------------
    # (2) SHORTEN DATASET LABELS
    # -----------------------------------------------------------
    df["dataset_short"] = (
        df["dataset"]
        .str.replace("_train_reduced", "", regex=False)
        .str.replace("_angular", "", regex=False)
        .str.replace("_euclidean", "", regex=False)
        .str.replace("_dot", "", regex=False)
    )

    # -----------------------------------------------------------
    # (3) WIDER FIGURE + (4) STRONGER GRIDLINES
    # -----------------------------------------------------------
    plt.figure(figsize=(20, 8))

    sns.violinplot(
        data=df,
        x="dataset_short",
        y="cluster_size",
        inner="box",
        density_norm="width",
        bw_adjust=1.5,
        cut=1,
        gridsize=200,
        linewidth=1.2
    )

    plt.yscale("log")
    plt.xticks(rotation=30, ha="right")
    plt.xlabel("Dataset", fontsize=15)
    plt.ylabel("Cluster Size (# of assigned vectors, log scale)", fontsize=15)
    plt.title(f"Distribution of Cluster Sizes Across Datasets ({K})", fontsize=18)

    plt.grid(axis="y", linestyle="--", alpha=0.75, linewidth=0.7)

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
