import superpos_functions as laser_utils
import numpy as np
import pandas as pd
import argparse
import configparser
import pickle as pkl
import matplotlib.pyplot as plt


colors= ['cornflowerblue','crimson','yellowgreen']

d_metrics = {}
def plot_triplet_waveforms(df, triplets, column_id, title, save_prefix, dt, vscale):
    """
    Plot waveforms for trial triplets with their average traces.

    Args:
        df (pd.DataFrame): DataFrame with 'Trial', 'Column_id', 'Waveforms', 'Type' columns.
        triplets (list[list[str]]): List of trial triplets as lists of strings or integers.
        column_id (int): Column ID to filter waveforms.
        title (str): Plot title.
        save_prefix (str): Path prefix for saving plot files.
    """
    cols = len(triplets) // 2
    fig, ax = plt.subplots(2, cols, figsize=(12, 8))
    ax = ax.flatten()

    fig_mean, ax_mean = plt.subplots(2, cols, figsize=(12, 8))
    ax_mean = ax_mean.flatten()

    for i, triplet in enumerate(triplets):
        for j, trial in enumerate(triplet):
            trial = int(trial)
            try:
                waveforms = df.loc[(df['Trial'] == trial) & 
                                   (df['Column_id'] == column_id),
                                   'Waveforms'].values[0]
            except IndexError:
                print(f"Trial {trial} not found.")
                continue

            # Assign label for plotting
            label = df.loc[df['Trial'] == trial, 'Type'].values[0] if j == 1 else 'control' if j == 0 else 'recovery'

            try:
                # Remove possible zero padded
                print("Fixing pad and v-scaling")
                waveforms = np.array(padded_to_min(waveforms))
                # Scale waveforms
                waveforms *= vscale

                print("Aligning and averaging")
                # Align waveforms by subtracting minimum value
                aligned_waveforms = np.array([w - min(w) for w in waveforms])
                # getting average
                w_mean = np.mean(aligned_waveforms, axis=0)
            except Exception as e:
                print(f"Error in trial {trial}: {e}")
                continue


            print("Getting time")
            w_time = np.arange(len(waveforms[0])) * dt

            print("Plotting")
            # Plot all aligned waveforms and their mean
            ax[i].plot(w_time, aligned_waveforms.T, color=colors[j], label=label, linewidth=0.01)
            ax[i].plot(w_time, w_mean.T, color=colors[j], label=label)
            ax_mean[i].plot(w_time, w_mean.T, color=colors[j], label=label)

            if j == 1:
                ax[i].set_title(label)
                ax_mean[i].set_title(label)
            
            ax[i].set_ylabel("mV")
            ax[i].set_xlabel("ms")

    fig.suptitle(title)
    fig_mean.suptitle(title)

    fig.tight_layout()
    fig_mean.tight_layout()

    fig.savefig(f'{save_prefix}_{title}.png', dpi=200, format='png')
    fig_mean.savefig(f'{save_prefix}_{title}_average.pdf', dpi=200, format='pdf')

    plt.show()

def clean_padded(waveforms):
    cleaned = []
    for waveform in waveforms:
        while len(waveform) > 0 and waveform[-1] == 0:
            waveform = waveform[:-1]
        cleaned.append(waveform)
    return cleaned

def padded_to_min(waveforms):
    cleaned = []
    for waveform in waveforms:
        i = len(waveform) - 1
        min = np.min(waveform)
        while i > 0 and waveform[i] == 0:
            waveform[i] = min
            i -= 1
        cleaned.append(waveform)
    return cleaned

def analyze_metrics(df, triplets, column_id, dt, vscale):
    labels = []
    for i, triplet in enumerate(triplets):
        for j, trial in enumerate(triplet):
            trial = int(trial)
            try:
                waveforms = df.loc[(df['Trial'] == trial) & 
                                (df['Column_id'] == column_id),
                                'Waveforms'].values[0]
            except IndexError:
                print(f"Trial {trial} not found.")
                continue
    
            # Assign label for plotting
            label = df.loc[df['Trial'] == trial, 'Type'].values[0] if j == 1 else 'control%d'%i if j == 0 else 'recovery%d'%i
            print(label)
            labels.append(label)
            try:
            # clean padded 0 in the end
            
                waveforms = clean_padded(waveforms)
                # include metrics in the dict
                d_metrics[label] = get_metrics(waveforms*vscale, dt)
            except Exception as e:
                print(f"Error in trial {trial}: {e}")
                continue    
   
    # Plot boxplots
    magnitudes = ['ms', 'mV', 'mV/ms', 'mV/ms']
    metrics_names = ['duration', 'amplitude', 'slope_dep', 'slope_rep']
    fig, ax = plt.subplots(ncols=4, figsize=(15, 5))
    colors = ['cornflowerblue', 'firebrick', 'olivedrab', 'gold']  # Define your colors here

    for i, metric_name in enumerate(metrics_names):
        metric_data = [d_metrics[label][metric_name] for label in labels]  # Order metrics by sorted labels
        bp = ax[i].boxplot(metric_data, labels=labels, patch_artist=True, showfliers=True)
        # set_box_colors(bp, colors[i % len(colors)])  # Cycle through colors
        ax[i].set_title(metric_name)
        ax[i].set_ylabel(magnitudes[i])
        # ax[i].tick_params(axis='x', rotation=45)
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()

    # print("Saving metrics at: ", img_name[:-4]+'metrics_boxplot.png')
    # print("Saving metrics at: ", img_name[:-4]+'metrics_boxplot.pdf')

    # plt.savefig(img_name[:-4] + 'metrics_boxplot.pdf', format='pdf')

    return d_metrics

import time
def get_metric_values(waveforms, fun, dt, thres_val):
    compute_metric = lambda w: fun(w, dt, thres_val=thres_val)

    values1 = []
    values2 = []
    w_time = 0

    for w in waveforms:
        if w.shape[0] == 0:
            continue
        # apply lambda function
        result = compute_metric(w)

        if isinstance(result, tuple):
            values1.append(result[0])
            values2.append(result[1])
        else:
            values1.append(result)
    
    return values1, values2


def get_metrics(waveforms, dt):
    d_metrics = {}
    d_metrics['duration'], _ = get_metric_values(waveforms, laser_utils.get_burst_duration_value, dt, thres_val=0.5)
    d_metrics['amplitude'], _ = get_metric_values(waveforms, laser_utils.get_spike_amplitude, dt, thres_val=0.5)
    d_metrics['slope_dep'], d_metrics['slope_rep'] = get_metric_values(waveforms, laser_utils.get_burst_slope, dt, thres_val=0.5)
    return d_metrics

def main(config_file_path, data_frame_path):
    config = configparser.ConfigParser()
    config.read(config_file_path)

    # Load dataframe
    df = pd.read_pickle(data_frame_path)

    # Parse triplets and column ID from config
    triplets = [triplet.split() for triplet in config['Superposition']['triplets'].split('|')]
    column_id = int(config['Superposition']['column_id'])

    # Extract a meaningful title from the file name
    title = data_frame_path[data_frame_path.rfind('/')+1:
                            data_frame_path.rfind('_extended_data.pkl')]
    
    save_prefix = config_file_path[:-4]

    # Waveform analysis
    time_step = df['Time'].iloc[0]  # e.g., 20000 Hz → 1/20000 s
    dt = time_step[1]


    # # Plot waveforms for triplets
    plot_triplet_waveforms(df, triplets, column_id, title, save_prefix, dt, vscale=1000)

    # d_metrics = analyze_metrics(df, triplets, column_id, dt, vscale=1000)
    # pd.DataFrame(d_metrics)

    # # pkl.save(d_metrics, f'{save_prefix}_metrics.pkl')
    



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
        help="Path to the dataframe extended."
    )

    # Parse the arguments
    args = parser.parse_args()

    # Example usage
    main(config_file_path=args.config_file_path, data_frame_path=args.dataframe_path)
