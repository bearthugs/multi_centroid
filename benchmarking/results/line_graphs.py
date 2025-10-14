import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# === Load data ===
df = pd.read_csv("results.csv")

# Remove "deep-image-96-angular" datasets
df = df[df["dataset"] != "deep-image-96-angular"]

# Create output folder
os.makedirs("line_graphs", exist_ok=True)

# Define unique datasets and M values
datasets = df["dataset"].unique()
M_values = sorted(df["M"].unique())
ef_values = sorted(df["efConstruction"].unique())
df["dim_by_vec"] = df["dimensions"] * df["no_of_vectors"]

def plot_by_dataset_for_fixed_ef(x_col, y_col, ef_value, title, filename, xlabel=None, ylabel=None):
    plt.figure(figsize=(10, 6))
    subset = df[df["efConstruction"] == ef_value]  # filter by efConstruction

    for dataset in datasets:
        data = subset[subset["dataset"] == dataset]
        grouped = data.groupby(x_col)[y_col].mean().reset_index()
        plt.plot(grouped[x_col], grouped[y_col], marker="o", label=dataset)

    plt.title(f"{title} (efConstruction={ef_value})")
    plt.xlabel(xlabel if xlabel else x_col)
    plt.ylabel(ylabel if ylabel else y_col)
    plt.legend(fontsize=7, ncol=2, title="Dataset")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join("line_graphs", filename))
    plt.close()

def plot_construction_vs_dimvec_all_M_for_fixed_ef(ef_value):
    plt.figure(figsize=(10, 6))
    subset = df[df["efConstruction"] == ef_value]
    M_values = sorted(subset["M"].unique())

    # Plot a line for each M value
    for M_val in M_values:
        grouped = (
            subset[subset["M"] == M_val]
            .groupby("dim_by_vec")["construction_time(ms)"]
            .mean()
            .reset_index()
        )
        plt.plot(
            grouped["dim_by_vec"],
            grouped["construction_time(ms)"],
            marker="o",
            label=f"M={M_val}"
        )

    plt.title(f"Construction Time vs (Dimensions × Number of Vectors)\n(efConstruction={ef_value})")
    plt.xlabel("Dimensions × Number of Vectors")
    plt.ylabel("Construction Time (ms)")
    plt.xscale("log")  # optional but usually helpful
    plt.legend(title="M", fontsize=8)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join("line_graphs", f"construction_vs_dimvec_allM_ef{ef_value}.png"))
    plt.close()


# Plot 1: M vs Construction Time (ms) for efConstruction = 100
plot_by_dataset_for_fixed_ef("M", "construction_time(ms)", 100,
    "M vs Construction Time (ms)", "M_vs_construction_time_ef100.png",
    "M", "Construction Time (ms)")

# Plot 2: M vs Construction Time (ms) for efConstruction = 200
plot_by_dataset_for_fixed_ef("M", "construction_time(ms)", 200,
    "M vs Construction Time (ms)", "M_vs_construction_time_ef200.png",
    "M", "Construction Time (ms)")

# Plot 3 and 4: Dimensions * No. of Vectors vs Construction Time
for ef in [100, 200]:
    plot_construction_vs_dimvec_all_M_for_fixed_ef(ef)

# Plot 5: M vs Query Time (ms) for efConstruction = 100
plot_by_dataset_for_fixed_ef("M", "query_time(ms)", 100,
    "M vs Query Time (ms)", "M_vs_query_time_ef100.png",
    "M", "Query Time (ms)")

# Plot 6: M vs Query Time (ms) for efConstruction = 200
plot_by_dataset_for_fixed_ef("M", "query_time(ms)", 200,
    "M vs Query Time (ms)", "M_vs_query_time_ef200.png",
    "M", "Query Time (ms)")



print("✅ All graphs generated and saved in the 'line_graphs' folder.")
