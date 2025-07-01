import superpos_functions as laser_utils
import h5py
import matplotlib.pyplot as plt
import numpy as np
import argparse
from scipy.signal import find_peaks
import os

dt = 0.1  # Sampling rate in ms
w_l = 300  # Left window
w_r = 500  # Right window
group_colors = {'control': 'cornflowerblue', 'laser': 'firebrick'}

def load_data(filepath):
    try:
        return h5py.File(filepath, 'r')
    except Exception as e:
        print("Error loading file:", e)
        return None

def classify_trial(data, key_col):
    try:
        shutter = data[:, key_col]
        if np.any(shutter > 1):
            return 'laser'
        else:
            return 'control'
    except Exception as e:
        print("Error classifying trial:", e)
        return None

def analyze_blocks_by_shutter(data, col_signal, key_col, dt, w_l, w_r):
    """
    Analiza bloques completos definidos por el estado del shutter (0 o 1) 
    y clasifica cada uno como 'laser' o 'control'.

    Parámetros:
    - data: ndarray de forma (n_samples, n_features)
    - col_signal: índice de la columna de la señal
    - key_col: índice de la columna del shutter (0 o 1)
    - dt: tamaño del paso temporal (en segundos)
    - w_l, w_r: muestras a izquierda y derecha del pico para extraer forma de onda
    """
    signal = data[:, col_signal] * 1000  # Convertir a mV
    shutter = data[:, key_col]

    n_samples = len(signal)
    waveforms_by_group = {'laser': [], 'control': []}

    # Detectar cambios de estado en el shutter
    transitions = np.where(np.diff(shutter.astype(int)) != 0)[0] + 1
    # Añadir inicio y fin como límites de bloques
    block_starts = np.concatenate(([0], transitions))
    block_ends = np.concatenate((transitions, [n_samples]))

    for start, end in zip(block_starts, block_ends):
        win_signal = signal[start:end]
        win_shutter = shutter[start:end]
        # fig, ax = plt.subplots(nrows=2)
        # ax[0].plot(win_signal)
        # ax[1].plot(win_shutter)
        # plt.show()
        group = 'laser' if np.mean(win_shutter) > 0.5 else 'control'

        amp = np.max(win_signal) - np.min(win_signal)
        if amp < 10:
            continue
        max_height = np.max(win_signal) - amp * 0.3
        
        spikes_t, _ = find_peaks(win_signal, height=max_height, distance=200)  # 1 s en muestras

        # fig, ax = plt.subplots(nrows=2)
        # ax[0].plot(win_signal)
        # ax[0].plot(spikes_t, np.ones(spikes_t.shape), 'x')
        # ax[1].plot(win_shutter)
        # plt.show()

        for spike in spikes_t[1:-1]:
            waveform = win_signal[max(0, spike - w_l):min(spike + w_r, len(win_signal))]
            # waveform -= waveform[0]
            waveforms_by_group[group].append(waveform)

        print(f"{group}: {len(waveforms_by_group[group])} spikes en bloque ({start*dt:.2f}s - {end*dt:.2f}s)")

    return waveforms_by_group


def get_metric_mean(waveforms, fun):
    values1 = []
    values2 = []
    for w in waveforms:
        try:
            value1, value2 = fun(w, dt)
            values1.append(value1)
            values2.append(value2)
        except:
            value1 = fun(w, dt)
            values1.append(value1)
    return np.mean(values1), np.mean(values2)

def get_metrics(waveforms):
    return {
        'duration': get_metric_mean(waveforms, laser_utils.get_spike_duration_value)[0],
        'amplitude': get_metric_mean(waveforms, laser_utils.get_spike_amplitude)[0],
        'slope_dep': get_metric_mean(waveforms, laser_utils.get_slope)[0],
        'slope_rep': get_metric_mean(waveforms, laser_utils.get_slope)[1]
    }

def plot_trial_waveforms(waveforms, group):
    time = np.arange(waveforms.shape[1]) * dt
    plt.plot(time, waveforms.T, color=group_colors[group], alpha=0.3, linewidth=0.4)
    plt.plot(time, np.mean(waveforms, axis=0), color=group_colors[group], linewidth=2, label=f"{group} mean")

def plot_summary_metrics(metrics_by_group):
    magnitudes = ['ms', 'mV', 'mV/ms', 'mV/ms']
    metric_names = ['duration', 'amplitude', 'slope_dep', 'slope_rep']
    fig, ax = plt.subplots(ncols=4, figsize=(15, 5))

    for i, name in enumerate(metric_names):
        for group, metrics in metrics_by_group.items():
            ax[i].scatter(group, metrics[name], color=group_colors[group])
        ax[i].set_title(name)
        ax[i].set_ylabel(magnitudes[i])
        ax[i].set_xticks(list(metrics_by_group.keys()))
        ax[i].set_xticklabels(list(metrics_by_group.keys()), rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required=True, help="Path to the file to analyze")
    ap.add_argument("-c", "--cols", required=False, default=0, type=int, help="Index of the signal channel")
    ap.add_argument("-t", "--trials", required=False, default=None, help="Specific trials to process")
    ap.add_argument("-on", "--output_name", required=False, default='', help="Base name for output files")
    ap.add_argument("-k", "--key_col", required=False, default=1, type=int, help="Column to check for laser/control classification")
    args = vars(ap.parse_args())

    filename = args['path']
    col_signal = args['cols']
    key_col = args['key_col']
    trials = list(map(int, args['trials'].split())) if args['trials'] else None
    out_base = args['output_name']
    output_dir = out_base + "_trials"
    os.makedirs(output_dir, exist_ok=True)

    f = load_data(filename)
    if f is None:
        return

    n_trial = 1
    
    metrics_by_group = {}

    while True:
        trial_name = f"Trial{n_trial}"
        struct = f"/{trial_name}/Synchronous Data/Channel Data"
        if trials and n_trial not in trials:
            n_trial += 1
            continue

        try:
            data = f[struct][()]
        except KeyError:
            break

        # try:
        #     print("Getting data from Trial",trial_name)
        #     waveforms_by_group = analyze_blocks_by_shutter(data, col_signal, key_col,dt,w_l,w_r)
        # except:
        #     print(f"Invalid channel in trial {n_trial}")
        #     n_trial += 1
        #     continue
        waveforms_by_group = analyze_blocks_by_shutter(data, col_signal, key_col, dt, w_l, w_r)

        plt.figure(figsize=(8, 5))
        plt.title(f"Trial {n_trial} - Waveforms by Group")

        for group in ['control', 'laser']:
            waveforms = np.array(waveforms_by_group[group])
            print("Analyzing group:", group)

            if len(waveforms) == 0:
                continue

            label = f"{group}_trial{n_trial}"
            metrics = get_metrics(waveforms)

            if group not in metrics_by_group:
                metrics_by_group[group] = []
            metrics_by_group[group].append(metrics)

            plot_trial_waveforms(waveforms, group)  # label no es necesario aquí

        plt.xlabel("ms")
        plt.ylabel("mV")
        plt.legend(title="Groups")
        plt.tight_layout()

        if output_dir:
            filename = os.path.join(output_dir, f"trial{n_trial}_waveforms.png")
            plt.savefig(filename, dpi=150, bbox_inches='tight')

        plt.show()
        plt.close()


        n_trial += 1
        if trials and n_trial > max(trials):
            break

    # Compute averages
    avg_metrics = {}
    for group, metrics_list in metrics_by_group.items():
        avg_metrics[group] = {}
        for key in metrics_list[0].keys():
            avg_metrics[group][key] = np.mean([m[key] for m in metrics_list])

    plot_summary_metrics(avg_metrics)
    # plot_difference_metrics(avg_metrics)

if __name__ == "__main__":
    main()
