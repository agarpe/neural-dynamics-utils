import superpos_functions as laser_utils
import numpy as np
import pandas as pd
import argparse
import configparser
import pickle as pkl
import matplotlib.pyplot as plt


colors= ['cornflowerblue','crimson','yellowgreen']

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
    d_metrics = {}
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

    df_metrics = pd.DataFrame(d_metrics)

    df_metrics = add_isi_metrics(df_metrics, df, triplets, column_id)

    return df_metrics


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


def add_isi_metrics(df_metrics, df_original, triplets, column_id):
    # Crear un diccionario para almacenar los nuevos datos por fila
    new_rows = {
        'isis': {},
        'n_spikes': {}
    }
    
    for i, triplet in enumerate(triplets):
        for j, trial in enumerate(triplet):
            isis = []
            n_spikes = []
            trial = int(trial)
            try:
                peaks_times = df_original.loc[(df_original['Trial'] == trial) & 
                                           (df_original['Column_id'] == column_id),
                                           'Peaks_Times'].values[0]
                burst_times = df_original.loc[(df_original['Trial'] == trial) & 
                                           (df_original['Column_id'] == column_id),
                                           'Bursts_Times'].values[0]
            except IndexError:
                print(f"Trial {trial} not found for ISI calculation.")
                continue
            
            label = df_original.loc[df_original['Trial'] == trial, 'Type'].values[0] if j == 1 else f'control{i}' if j == 0 else f'recovery{i}'
            
            for burst in burst_times:
                spikes = peaks_times[np.where(peaks_times > burst[0])]
                spikes = spikes[np.where(spikes < burst[1])]
                isis.append(np.mean(np.diff(spikes)))
                n_spikes.append(len(spikes))
            
            if len(peaks_times) > 1:
                new_rows['isis'][label] = isis
                new_rows['n_spikes'][label] = n_spikes
            else:
                new_rows['isis'][label] = []
                new_rows['n_spikes'][label] = 0
    
    # Crear nuevos DataFrames para cada métrica y concatenarlos verticalmente
    df_isi = pd.DataFrame([new_rows['isis']], index=['isis'])
    df_n_spikes = pd.DataFrame([new_rows['n_spikes']], index=['n_spikes'])
    print(df_metrics)
    print(df_isi)
    print(df_n_spikes)
    # Concatenar con el DataFrame original
    return pd.concat([df_metrics, df_isi, df_n_spikes])


def plot_metrics(df_metrics, save_prefix, selected_trials=None):

    # Plot boxplots
    # magnitudes = ['ms', 'mV', 'mV/ms', 'mV/ms']
    # metrics_names = ['duration', 'amplitude', 'slope_dep', 'slope_rep']
    # colors = ['cornflowerblue', 'firebrick', 'olivedrab', 'gold']  # Define your colors here

    metrics = df_metrics.index
    all_trials = df_metrics.columns

    # Usa solo los trials seleccionados para el primer plot
    trials = selected_trials if selected_trials is not None else all_trials

    # ---------- Plot 1: un subplot por métrica ----------
    fig, axs = plt.subplots(len(metrics), 1, figsize=(12, 10), sharex=True)

    # Define the color cycle
    if selected_trials is None:
        colors = ['cornflowerblue', 'crimson', 'yellowgreen']
    else:
        # Get default matplotlib color cycle
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']

    for i, metric in enumerate(metrics):
        ax = axs[i]
        
        data = [df_metrics.loc[metric, col] for col in trials]
        
        # Create boxplot with specified colors
        boxplot = ax.boxplot(data, positions=range(len(trials)), widths=0.5, patch_artist=True)
        
        # Set colors for boxes
        for j, box in enumerate(boxplot['boxes']):
            box.set_facecolor(colors[j % len(colors)])
        
        # Puntos con jitter with matching colors
        for j, points in enumerate(data):
            jitter = np.random.normal(0, 0.05, size=len(points))
            ax.plot(np.full(len(points), j) + jitter, points, 'o', alpha=0.4, 
                    markersize=4, color=colors[j % len(colors)])
        
        ax.set_title(metric)
        ax.set_xticks(range(len(trials)))
        ax.set_xticklabels(trials, rotation=45)

    fig.suptitle("Metrics per trial", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if selected_trials is None:
        fig.savefig(f"{save_prefix}_metrics_all.png", format='png', dpi=200)
    else:
        fig.savefig(f"{save_prefix}_metrics_lasers.png", format='png', dpi=200)
    plt.show()

RED = '\033[91m'
RESET = '\033[0m'
   

def main(config_file_path, data_frame_path):
    config = configparser.ConfigParser()
    config.read(config_file_path)

    # Load dataframe
    df = pd.read_pickle(data_frame_path)

    # Parse triplets and column ID from config
    triplets = [triplet.split() for triplet in config['Superposition']['triplets'].split('|')]
    column_id = int(config['Superposition']['column_id'])
    compute_metrics = config['Superposition']['compute_metrics'].lower() == 'y'

    # Extract a meaningful title from the file name
    title = data_frame_path[data_frame_path.rfind('/')+1:
                            data_frame_path.rfind('_extended_data.pkl')]
    
    save_prefix = config_file_path[:-4]
    

    # Waveform analysis
    time_step = df['Time'].iloc[0]  # e.g., 20000 Hz → 1/20000 s
    dt = time_step[1]

    file_name = f'{save_prefix}_waveform_metrics.pkl'

    try:
        df_metrics = pd.read_pickle(f'{file_name}')
        print(f"Warning: Metrics file already exists")

        print(df_metrics)

        overwrite = input("Do you want to overwrite it? (y/n): ").lower()
        if overwrite == 'y':
            print(f"Warning: Recomputing metrics and overwriting file")
            df_metrics = analyze_metrics(df, triplets, column_id, dt, vscale=1000)
            df_metrics.to_pickle(f'{file_name}')
        else:
            exit()
    except FileNotFoundError:
        print(f"Warning: Metrics file not found, analyzing waveforms...")
        df_metrics = analyze_metrics(df, triplets, column_id, dt, vscale=1000)
        print(f"Saving metrics dataframe {file_name}")
        df_metrics.to_pickle(f'{file_name}')

    print(df_metrics.columns)

    plot_metrics(df_metrics, save_prefix)
    
    excluded = [col for col in df_metrics.columns if not any(k in col for k in ['control', 'recovery'])]
    print(excluded)

    plot_metrics(df_metrics, save_prefix, excluded)


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
