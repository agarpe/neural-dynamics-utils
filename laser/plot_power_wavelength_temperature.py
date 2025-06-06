
import superpos_functions as laser_utils
import numpy as np
import pandas as pd
import argparse
import configparser
import pickle as pkl
import matplotlib.pyplot as plt

def plot_parameter_metrics(df_metrics, powers, wavelengths, temperatures, trial_numbers, save_prefix):
    """
    df_metrics: DataFrame with rows=metrics, cols=columns like control0, param, recovery0 ...
    config: dict with keys 'powers', 'wavelength', 'temperature' as space-separated strings
    trial_numbers: list of ints, e.g. [2,4,6,8,10,12]
    """
    metrics = df_metrics.index.tolist()
    columns = df_metrics.columns.tolist()
    
    selected_cols = [col for col in df_metrics.columns if any(str(t) in col for t in trial_numbers) and not col.startswith('control') and not col.startswith('recovery')]

    # Dictionary of parameter values for selected trials
    param_values = {
        'power': powers,
        'wavelength': wavelengths,
        'temperature': temperatures,
    }

    # --- Plot 1: boxplots per metric, grouped by parameter values ---
    for metric in metrics:
        fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
        fig.suptitle(f"Metric: {metric} boxplot grouped by parameters")
        
        metric_data = []
        for col in selected_cols:
            vals = df_metrics.loc[metric, col]
            # ensure list or array
            metric_data.append(vals if isinstance(vals, (list, np.ndarray)) else [vals])

        # For each parameter
        for i, (param_name, values) in enumerate(param_values.items()):
            ax = axs[i]
            cmap = plt.get_cmap('tab10')
            colors = [cmap(j % 10) for j in range(len(values))]

            xtick_locs = []
            xtick_labels = []
            used_positions = set()

            for j, (val, col) in enumerate(zip(values, selected_cols)):
                if np.isnan(val):
                    continue

                y = df_metrics.loc[metric, col]
                color = colors[j]

                # Jitter position
                jitter_x = np.random.normal(0, 2)
                pos = val + jitter_x

                # Boxplot
                ax.boxplot([y], positions=[pos], widths=5,
                        patch_artist=True,
                        boxprops=dict(facecolor=color, color='black'),
                        medianprops=dict(color='black'),
                        whiskerprops=dict(color='black'),
                        capprops=dict(color='black'),
                        flierprops=dict(markerfacecolor=color, marker='o', alpha=0.3))

                # Scatter
                scatter_jitter = np.random.normal(0, 0.5, size=len(y))
                ax.scatter(np.full(len(y), pos) + scatter_jitter, y, s=10, alpha=0.6, color=color)

                # Legend
                ax.plot([], [], color=color, label=col)  # dummy for legend

                # Only add xtick label once per unique param value
                if val not in used_positions:
                    xtick_locs.append(val)
                    xtick_labels.append(str(val))
                    used_positions.add(val)

            ax.set_title(param_name)
            ax.set_xlabel(param_name)
            if i == 0:
                ax.set_ylabel(metric)
            ax.set_xticks(xtick_locs)
            ax.set_xticklabels(xtick_labels)
            ax.grid(True)
            ax.legend(fontsize=8, loc='best')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        print("Saving fig", f"{save_prefix}_parameters_{metric}.png")
        fig.savefig(f"{save_prefix}_parameters_{metric}.png",format='png', dpi=150)
        # plt.show()


        
    # --- Plot 2: mean metric values per trial vs parameters with dual y-axis ---
    for metric in metrics:
        fig, ax1 = plt.subplots(figsize=(8, 6))
        fig.suptitle(f"Metric: {metric} mean vs parameters")

        # Get means per trial for selected columns
        means = []
        for col in selected_cols:
            vals = df_metrics.loc[metric, col]
            vals_flat = vals if isinstance(vals, (list, np.ndarray)) else [vals]
            means.append(np.nanmean(vals_flat))

        x = trial_numbers
        
        # Plot power on left y-axis
        color1 = 'tab:blue'
        ax1.set_xlabel('Trial Number')
        ax1.set_ylabel('Power', color=color1)
        ax1.scatter(x, param_values['power'], color=color1, label='Power', marker='o')
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.grid(True)

        # Plot wavelength on right y-axis
        ax2 = ax1.twinx()
        color2 = 'tab:green'
        ax2.set_ylabel('Wavelength', color=color2)
        ax2.scatter(x, param_values['wavelength'], color=color2, label='Wavelength', marker='^')
        ax2.tick_params(axis='y', labelcolor=color2)

        # Optional: plot temperature on third axis? matplotlib does not support triple y-axis well, so skip here

        # Plot mean metric values on bottom x axis, for clarity plot as a line on secondary x axis
        ax3 = ax1.twiny()
        color3 = 'tab:red'
        ax3.set_xlim(ax1.get_xlim())
        ax3.set_xticks(x)
        ax3.set_xticklabels([f"{m:.2f}" for m in means], rotation=45)
        ax3.set_xlabel(f'Mean {metric} per Trial', color=color3)
        ax3.tick_params(axis='x', labelcolor=color3)

        plt.tight_layout()
        print("Saving fig", f"{save_prefix}_parameters_scatter_{metric}.png")
        fig.savefig(f"{save_prefix}_parameters_scatter_{metric}.png",format='png', dpi=150)
        # plt.show()



def main(config_file_path, data_frame_path):
    config = configparser.ConfigParser()
    config.read(config_file_path)

    # Load dataframe
    df = pd.read_pickle(data_frame_path)

    # Parse triplets and column ID from config
    trials = [int(trial) for trial in config['Power-wavelengths']['laser_trials'].split()]
    powers = [float(power) for power in config['Power-wavelengths']['powers'].split()]
    wavelengths = [float(wl) for wl in config['Power-wavelengths']['wavelengths'].split()]
    temperatures = [float(temp) for temp in config['Power-wavelengths']['temperatures'].split()]
    

    # Extract a meaningful title from the file name
    title = data_frame_path[data_frame_path.rfind('/')+1:
                            data_frame_path.rfind('_extended_data.pkl')]
    
    save_prefix = config_file_path[:-4]
    

    try:
        df_metrics = pd.read_pickle(data_frame_path)
        print(f"Warning: Loading metrics dataframe from {data_frame_path}")
    except:
        print("Warning: metrics file not found, run superpos_burst_waveform first...")

    print(df_metrics.columns)

    plot_parameter_metrics(df_metrics, powers, wavelengths, temperatures, trials, save_prefix)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process an H5 file and a config file.")


    # Define the arguments
    parser.add_argument(
        "config_file_path",
        type=str,
        help="Path to the config (INI) file."
    )

    # Define the arguments
    parser.add_argument(
        "dataframe_path",
        type=str,
        help="Path to the metrics dataframe."
    )

    args = parser.parse_args()

    # Example usage
    main(config_file_path=args.config_file_path, data_frame_path=args.dataframe_path)
