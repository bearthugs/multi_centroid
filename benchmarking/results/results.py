import os
import pandas as pd
import matplotlib.pyplot as plt

# === Load data ===
df = pd.read_csv("results.csv")

# Remove "deep-image-96-angular" datasets
df = df[df["dataset"] != "deep-image-96-angular"]

# Create output folder
os.makedirs("graphs", exist_ok=True)

# Define unique datasets and M values
datasets = df["dataset"].unique()
M_values = sorted(df["M"].unique())
ef_values = sorted(df["efConstruction"].unique())
df["dim_to_vec_ratio"] = df["dimensions"] * df["no_of_vectors"]

# === Helper: Plot function for dataset-based plots ===
def plot_by_dataset(x_col, y_col, title, filename, xlabel=None, ylabel=None):
    plt.figure(figsize=(10, 6))
    for dataset in datasets:
        data = df[df["dataset"] == dataset]
        grouped = data.groupby(x_col)[y_col].mean().reset_index()
        plt.plot(grouped[x_col], grouped[y_col], marker="o", label=dataset)

    plt.title(title)
    plt.xlabel(xlabel if xlabel else x_col)
    plt.ylabel(ylabel if ylabel else y_col)
    plt.legend(fontsize=7, ncol=2)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join("graphs", filename))
    plt.close()

# === Helper: Plot function for M-based plots at fixed efConstruction ===
def plot_by_M_for_fixed_ef(x_col, y_col, ef_value, title, filename, xlabel=None, ylabel=None):
    plt.figure(figsize=(10, 6))
    subset = df[df["efConstruction"] == ef_value]
    for M in M_values:
        data = subset[subset["M"] == M]
        grouped = data.groupby(x_col)[y_col].mean().reset_index()
        plt.plot(grouped[x_col], grouped[y_col], marker="o", label=f"M={M}")

    plt.title(title)
    plt.xlabel(xlabel if xlabel else x_col)
    plt.ylabel(ylabel if ylabel else y_col)
    plt.legend(title="M", fontsize=8)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join("graphs", filename))
    plt.close()

# === 1–6: Dataset-based plots ===
plot_by_dataset("M", "construction_time(ms)", "M vs Construction Time (ms)", "M_vs_construction_time.png", "M", "Construction Time (ms)")
plot_by_dataset("efConstruction", "construction_time(ms)", "efConstruction vs Construction Time (ms)", "efConstruction_vs_construction_time.png", "efConstruction", "Construction Time (ms)")
plot_by_dataset("M", "query_time(ms)", "M vs Query Time (ms)", "M_vs_query_time.png", "M", "Query Time (ms)")
plot_by_dataset("efConstruction", "query_time(ms)", "efConstruction vs Query Time (ms)", "efConstruction_vs_query_time.png", "efConstruction", "Query Time (ms)")
plot_by_dataset("M", "recall@100", "M vs Recall@100", "M_vs_recall.png", "M", "Recall@100")
plot_by_dataset("efConstruction", "recall@100", "efConstruction vs Recall@100", "efConstruction_vs_recall.png", "efConstruction", "Recall@100")


# 7 Recall vs dim:no_of_vectors (ef=200)
plot_by_M_for_fixed_ef(
    "dim_to_vec_ratio",
    "recall@100",
    200,
    "Recall@100 vs (Dimensions / No. of Vectors) for efConstruction=200",
    "recall_vs_dim_ratio_ef200.png",
    "Dimensions : No. of Vectors Ratio",
    "Recall@100"
)

# 8 Recall vs dim:no_of_vectors (ef=100)
plot_by_M_for_fixed_ef(
    "dim_to_vec_ratio",
    "recall@100",
    100,
    "Recall@100 vs (Dimensions / No. of Vectors) for efConstruction=100",
    "recall_vs_dim_ratio_ef100.png",
    "Dimensions : No. of Vectors Ratio",
    "Recall@100"
)

# 9 Construction Time vs dim:no_of_vectors (ef=200)
plot_by_M_for_fixed_ef(
    "dim_to_vec_ratio",
    "construction_time(ms)",
    200,
    "Construction Time vs (Dimensions / No. of Vectors) for efConstruction=200",
    "construction_time_vs_dim_ratio_ef200.png",
    "Dimensions : No. of Vectors Ratio",
    "Construction Time (ms)"
)

# 10 Construction Time vs dim:no_of_vectors (ef=100)
plot_by_M_for_fixed_ef(
    "dim_to_vec_ratio",
    "construction_time(ms)",
    100,
    "Construction Time vs (Dimensions / No. of Vectors) for efConstruction=100",
    "construction_time_vs_dim_ratio_ef100.png",
    "Dimensions : No. of Vectors Ratio",
    "Construction Time (ms)"
)

# 11 Query Time vs dim:no_of_vectors (ef=100)
plot_by_M_for_fixed_ef(
    "dim_to_vec_ratio",
    "query_time(ms)",
    100,
    "Query Time vs (Dimensions / No. of Vectors) for efConstruction=100",
    "query_time_vs_dim_ratio_ef100.png",
    "Dimensions : No. of Vectors Ratio",
    "Query Time (ms)"
)

# 12 Query Time vs dim:no_of_vectors (ef=200)
plot_by_M_for_fixed_ef(
    "dim_to_vec_ratio",
    "query_time(ms)",
    200,
    "Query Time vs (Dimensions / No. of Vectors) for efConstruction=200",
    "query_time_vs_dim_ratio_ef200.png",
    "Dimensions : No. of Vectors Ratio",
    "Query Time (ms)"
)

print("✅ All graphs generated and saved in the 'graphs' folder.")
