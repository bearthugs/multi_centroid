import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_gmm_training_graphs(csv_path='training_summary_attempt2.csv'):
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
        "font.size": 15,
        "axes.titlesize": 22,
        "axes.labelsize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 14
    })

    # Helper: legend at bottom for all plots
    def bottom_legend():
        plt.legend(
            loc='upper center',
            bbox_to_anchor=(0.5, -0.18),  # move further down if needed
            ncol=3,
            frameon=False
        )

    # Generic plotting function
    def plot_generic(x_col, y_col, xlabel, ylabel, title, filename, yscale=None):
        fig, ax = plt.subplots()

        # Distinct marker + line style cycles
        markers = ['o', 's', '^', 'D', 'P', '*', 'X', 'v', '<', '>', 'H']
        linestyles = ['-', '--', '-.', ':']
        
        datasets = list(df['dataset'].unique())
        style_map = {}
        for i, dataset in enumerate(datasets):
            marker = markers[i % len(markers)]
            linestyle = linestyles[i % len(linestyles)]
            style_map[dataset] = (marker, linestyle)

        for dataset, group in df.groupby('dataset'):
            group = group.sort_values(by=x_col)
            marker, linestyle = style_map[dataset]

            ax.plot(
                group[x_col],
                group[y_col],
                marker=marker,
                linestyle=linestyle,
                linewidth=2.0,
                markersize=8,
                label=dataset
            )
        
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if yscale:
            ax.set_yscale(yscale)

        bottom_legend()

        fig.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {os.path.join(output_dir, filename)}")


    # === 1. K vs Runtime (in ms) ===
    plot_generic(
        x_col='K',
        y_col='time_ms',
        xlabel='Number of Clusters (K)',
        ylabel='Runtime (ms)',
        title='K vs Construction Time (All Datasets)',
        filename='k_vs_runtime_ms.png',
        yscale='log'
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
    plot_gmm_training_graphs("training_summary_attempt2.csv")
