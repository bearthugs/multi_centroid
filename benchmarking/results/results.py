import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Settings for nicer plots
sns.set(style="whitegrid")
plt.rcParams.update({'figure.max_open_warning': 0})

# Load CSV
df = pd.read_csv("results.csv")

# Ensure directory for graphs exists
output_dir = "graphs"
os.makedirs(output_dir, exist_ok=True)

datasets = df["dataset"].unique()

for dataset in datasets:
    data = df[df["dataset"] == dataset]

    # Plot: Recall@100 vs. M for different efConstruction
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=data, x="M", y="recall@100", hue="efConstruction", marker="o")
    plt.title(f"Recall@100 vs. M — {dataset}")
    plt.xlabel("M")
    plt.ylabel("Recall@100")
    plt.legend(title="efConstruction")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{dataset}_recall_vs_M.png")
    plt.close()

    # Plot: Construction time vs. M for different efConstruction
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=data, x="M", y="construction_time(ms)", hue="efConstruction", marker="o")
    plt.title(f"Construction Time vs. M — {dataset}")
    plt.xlabel("M")
    plt.ylabel("Construction Time (ms)")
    plt.legend(title="efConstruction")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{dataset}_construction_time_vs_M.png")
    plt.close()

    # Plot: Query time vs. M for different efConstruction
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=data, x="M", y="query_time(ms)", hue="efConstruction", marker="o")
    plt.title(f"Query Time vs. M — {dataset}")
    plt.xlabel("M")
    plt.ylabel("Query Time (ms)")
    plt.legend(title="efConstruction")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{dataset}_query_time_vs_M.png")
    plt.close()

    # Plot: Recall vs. Construction time
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=data, x="construction_time(ms)", y="recall@100", hue="M", style="efConstruction", s=100)
    plt.title(f"Recall@100 vs. Construction Time — {dataset}")
    plt.xlabel("Construction Time (ms)")
    plt.ylabel("Recall@100")
    plt.legend(title="M / efConstruction", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{dataset}_recall_vs_construction_time.png")
    plt.close()


# ====== Averages across all datasets ======
numeric_cols = df.select_dtypes(include='number').columns

avg_df = df.groupby(["M", "efConstruction"], as_index=False)[numeric_cols].mean()


# Plot: Average Recall@100 vs. M
plt.figure(figsize=(10, 6))
sns.lineplot(data=avg_df, x="M", y="recall@100", hue="efConstruction", marker="o")
plt.title("Average Recall@100 vs. M (Across All Datasets)")
plt.xlabel("M")
plt.ylabel("Average Recall@100")
plt.legend(title="efConstruction")
plt.tight_layout()
plt.savefig(f"{output_dir}/average_recall_vs_M.png")
plt.close()

# Plot: Average Construction Time vs. M
plt.figure(figsize=(10, 6))
sns.lineplot(data=avg_df, x="M", y="construction_time(ms)", hue="efConstruction", marker="o")
plt.title("Average Construction Time vs. M (Across All Datasets)")
plt.xlabel("M")
plt.ylabel("Average Construction Time (ms)")
plt.legend(title="efConstruction")
plt.tight_layout()
plt.savefig(f"{output_dir}/average_construction_time_vs_M.png")
plt.close()

# Plot: Average Query Time vs. M
plt.figure(figsize=(10, 6))
sns.lineplot(data=avg_df, x="M", y="query_time(ms)", hue="efConstruction", marker="o")
plt.title("Average Query Time vs. M (Across All Datasets)")
plt.xlabel("M")
plt.ylabel("Average Query Time (ms)")
plt.legend(title="efConstruction")
plt.tight_layout()
plt.savefig(f"{output_dir}/average_query_time_vs_M.png")
plt.close()

# Plot: Average Recall vs. Average Construction Time
plt.figure(figsize=(10, 6))
sns.scatterplot(data=avg_df, x="construction_time(ms)", y="recall@100", hue="M", style="efConstruction", s=100)
plt.title("Average Recall@100 vs. Average Construction Time (Across All Datasets)")
plt.xlabel("Average Construction Time (ms)")
plt.ylabel("Average Recall@100")
plt.legend(title="M / efConstruction", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig(f"{output_dir}/average_recall_vs_construction_time.png")
plt.close()

print(f"Graphs saved in folder '{output_dir}'")
