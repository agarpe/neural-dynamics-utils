
import superpos_functions as laser_utils
import numpy as np
import pandas as pd
import argparse
import configparser
import pickle as pkl
import matplotlib.pyplot as plt

def plot_parameter_metrics_heatmap(df_metrics, powers, wavelengths, temperatures, trial_numbers, save_prefix, parameter='temperature'):
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
    # --- Plot: color-coded mean of metric vs (power, wavelength) ---
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.suptitle(f"Mean {metric} vs {parameter} and Wavelength")

        powers_list = param_values[parameter]
        wavelengths_list = param_values['wavelength']

        means = []
        for col in selected_cols:
            vals = df_metrics.loc[metric, col]
            vals_flat = vals if isinstance(vals, (list, np.ndarray)) else [vals]
            means.append(np.nanmean(vals_flat))

        scatter = ax.scatter(
            wavelengths_list,
            means,
            c=powers_list,
            cmap='hot_r',
            s=100,
            edgecolors='k'
        )

        metric_label = f'{metric}: abs(laser - mean(control)'

        ax.set_xlabel('Wavelength')
        ax.set_ylabel(metric_label)
        ax.grid(True)

        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(parameter)

        plt.tight_layout()
        save_path = f"{save_prefix}_{parameter}_heatmap_{metric}.png"
        print("Saving fig", save_path)
        fig.savefig(save_path, format='png', dpi=150)
        # plt.show()


def get_differences(df_metrics):
    # Prepare the new DataFrame to hold the differences
    df_diff = pd.DataFrame(index=df_metrics.index)

    # Iterate over columns in steps of 3: [control, stim, recovery]
    columns = df_metrics.columns
    i = 0
    while i < len(columns) - 1:
        col_control = columns[i]
        col_stim = columns[i + 1]
        
        # Ensure the second column is a stimulation and not a recovery
        if 'recovery' not in col_stim and 'control' in col_control:
            diff_col_name = f"{col_stim}-diff"
            # Subtract element-wise for each metric
            # df_diff[diff_col_name] = abs(df_metrics[col_stim].apply(np.nanmean) - df_metrics[col_control].apply(np.nanmean))

            #  For each metric (row), subtract mean of control from each value in stim
            df_diff[diff_col_name] = df_metrics.apply(
                lambda row: abs(np.array(row[col_stim]) - np.nanmean(row[col_control])),
                axis=1
            )
            i += 3  # skip recovery
        else:
            i += 1  # move forward if structure doesn't match

    # Done
    print(df_diff.head())
    return df_diff


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

    df_diff = get_differences(df_metrics)

    plot_parameter_metrics_heatmap(df_diff, powers, wavelengths, temperatures, trials, save_prefix, parameter='power')
    plot_parameter_metrics_heatmap(df_diff, powers, wavelengths, temperatures, trials, save_prefix, parameter='temperature')



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
