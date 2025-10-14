import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === Load data ===
df = pd.read_csv("results.csv")

# Remove unwanted datasets
df = df[df["dataset"] != "deep-image-96-angular"]

# Create derived column
df["dim_by_vec"] = df["dimensions"] * df["no_of_vectors"]

# Create output folder
os.makedirs("bar_graphs", exist_ok=True)

# Unique M values and efConstruction values
M_values = sorted(df["M"].unique())
ef_values = sorted(df["efConstruction"].unique())

for ef in ef_values:
    subset_ef = df[df["efConstruction"] == ef]

    for M_val in M_values:
        subset = subset_ef[subset_ef["M"] == M_val]

        # Sort by dim_by_vec in ascending order
        subset = subset.sort_values("dim_by_vec")

        # Create y-axis labels with dataset name + dim*vec in brackets
        y_labels = [
            f"{row['dataset']} ({row['dim_by_vec']})" 
            for _, row in subset.iterrows()
        ]

        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=subset,
            y=y_labels,
            x="construction_time(ms)",
            palette="tab10"
        )

        plt.xscale("log")  # log scale for construction time
        plt.xlabel("Construction Time (ms) [log scale]")
        plt.ylabel("Dataset (dim*vec)")
        plt.title(f"Construction Time by Dataset\nM={M_val}, efConstruction={ef}")
        plt.grid(True, linestyle='--', alpha=0.5, axis='x')
        plt.tight_layout()

        # Save figure
        plt.savefig(f"bar_graphs/construction_time_dataset_M{M_val}_ef{ef}.png")
        plt.close()

print("✅ Horizontal bar plots ordered by ascending dim*vec saved for each M and efConstruction.")
