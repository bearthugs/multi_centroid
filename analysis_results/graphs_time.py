import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_gmm_training_graphs(csv_path='training_summary.csv'):
    """Generate GMM training summary graphs for multiple datasets (time in ms)."""
    output_dir = "gmm_graphs"
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    df['n_dim'] = df['n'] * df['dim']  # derived column

    # Convert runtime from seconds → milliseconds
    df['time_ms'] = df['time_sec'] * 1000

    # Shared styling
    plt.rcParams.update({
        "figure.figsize": (12, 8),
        "axes.grid": True,
        "grid.alpha": 0.7,
        "grid.linestyle": "--",
        "font.size": 13
    })

    def plot_generic(x_col, y_col, xlabel, ylabel, title, filename, yscale=None):
        plt.figure()
        for dataset, group in df.groupby('dataset'):
            group = group.sort_values(by=x_col)
            plt.plot(group[x_col], group[y_col], marker='o', label=dataset)
        
        plt.title(title, fontsize=16)
        plt.xlabel(xlabel, fontsize=14)
        plt.ylabel(ylabel, fontsize=14)
        if yscale:
            plt.yscale(yscale)
        plt.legend(ncol=2, fontsize=10)
        plt.tight_layout()
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300)
        plt.close()
        print(f"Saved: {filepath}")

    # === 1. K vs Runtime (in ms) ===
    plot_generic(
        x_col='K',
        y_col='time_ms',
        xlabel='Number of Clusters (K)',
        ylabel='Runtime (milliseconds)',
        title='K vs Runtime (All Datasets)',
        filename='k_vs_runtime_ms.png',
        yscale='log'  # runtimes vary widely
    )

    # === 2. K vs Average Log-Likelihood ===
    plot_generic(
        x_col='K',
        y_col='avg_loglike',
        xlabel='Number of Clusters (K)',
        ylabel='Average Log-Likelihood',
        title='K vs Average Log-Likelihood (All Datasets)',
        filename='k_vs_loglike.png'
    )

if __name__ == "__main__":
    plot_gmm_training_graphs("training_summary.csv")
