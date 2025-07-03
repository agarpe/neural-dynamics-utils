"""
Copyright (c) 2025 Grupo de Neurocomputación Biológica. Universidad Autónoma de Madrid
Authors:
    Alicia Garrido Peña
    Pablo Sánchez

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above
      copyright notice, this list of conditions and the following
      disclaimer in the documentation and/or other materials provided
      with the distribution.
    * Neither the name of the author nor the names of his contributors
      may be used to endorse or promote products derived from this
      software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import h5py
import pandas as pd
import configparser
import argparse
import pickle


def FIR(neuron_signal, is_lowpass, cutoff, sampling_rate = 10000):

    nyquist = 0.5 * sampling_rate  # Nyquist frequency

    # Design the FIR filter using the window method
    num_taps = 101  # Number of taps (filter order + 1)
    fir_coeff = signal.firwin(num_taps, cutoff/nyquist, pass_zero=is_lowpass)

    # Apply the high-pass FIR filter using filtfilt for zero-phase distortion
    signal_filtered = signal.filtfilt(fir_coeff, 1.0, neuron_signal)

    # Plot the original signal and the filtered signal
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(neuron_signal, label='neuron signal')
    plt.title('neuron signal')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(signal_filtered, label='Filtered Signal', color='r')
    plt.title('Filtered Signal (Lowpass 100 Hz FIR Filter)')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    return signal_filtered

def get_peaks(neuron_signal, percentage_threshold, min_distance_ms, sampling_rate):
    min_distance = min_distance_ms / sampling_rate
    # Calculate range of the signal
    # TODO  control outliers
    signal_range = abs(max(neuron_signal) - min(neuron_signal))

    # Calculate the absolute threshold with respect to the minimum
    absolute_threshold = min(neuron_signal) + percentage_threshold * signal_range

    peaks, _ = signal.find_peaks(neuron_signal, height=absolute_threshold, distance=min_distance, prominence=0.05*signal_range)

    return peaks, absolute_threshold

def detect_bursts_from_spikes(
    spike_indices, 
    sampling_rate,  # Firing rate in Hz
    min_spikes=3, 
    min_spike_dist=1, 
    max_spike_dist=20, 
    min_burst_dist=40
):
    """
    Detect bursts from pre-detected spike indices based on spike distances and burst characteristics.

    Parameters:
    - spike_indices: List or array of spike indices (already detected spikes).
    - firing_rate: Firing rate of the signal in Hz.
    - min_spikes: Minimum number of spikes to consider a burst.
    - min_spike_dist: Minimum distance (in ms) between spikes within a burst.
    - max_spike_dist: Maximum distance (in ms) between spikes within a burst.
    - min_burst_dist: Minimum distance (in ms) between bursts.

    Returns:
    - bursts: List of bursts, each burst is a list of spike indices.
    """
    # Convert time in ms to "points" based on the firing rate
    ms_to_points = lambda ms: int(ms / sampling_rate)
    
    min_spike_dist = ms_to_points(min_spike_dist)
    max_spike_dist = ms_to_points(max_spike_dist)
    min_burst_dist = ms_to_points(min_burst_dist)


    bursts = []  # List to store detected bursts
    current_burst = []  # Temporary list for the current burst

    # Step 1: Group spikes into bursts based on distance criteria
    for i in range(1, len(spike_indices)):
        # Check the distance between consecutive spikes
        spike_distance = spike_indices[i] - spike_indices[i - 1]

        # If the distance is within the burst limits, add spike to the current burst
        if min_spike_dist <= spike_distance <= max_spike_dist:
            if not current_burst:
                current_burst.append(spike_indices[i - 1])  # Add the previous spike if starting a new burst
            current_burst.append(spike_indices[i])

        # If the distance exceeds the maximum allowed between spikes, finalize the current burst
        else:
            if len(current_burst) >= min_spikes:
                bursts.append(current_burst)
            current_burst = [spike_indices[i]]  # Start a new burst

    # Finalize the last burst if it meets the criteria
    if len(current_burst) >= min_spikes:
        bursts.append(current_burst)

    # Step 2: Filter bursts based on minimum distance between bursts
    filtered_bursts = []
    last_burst_end = -min_burst_dist  # Ensure the first burst starts at the beginning

    for burst in bursts:
        burst_start = burst[0]
        burst_end = burst[-1]

        # Ensure bursts are sufficiently far apart
        if burst_start - last_burst_end >= min_burst_dist:
            filtered_bursts.append(burst)
            last_burst_end = burst_end  # Update the last burst end position

    return filtered_bursts


# Helper function for plotting
def plot_signal_with_peaks_and_bursts(ax, v_signal, time, peaks, peaks_time, bursts, absolute_threshold):
    # plt.figure(figsize=(15, 6))

    # Plot the full signal
    ax.plot(time, v_signal, label='Signal', color='blue', linewidth=1)
    
    # Plot threshold
    ax.hlines(y=absolute_threshold, xmin=0, xmax=time[-1], color='g', linestyle='--', label='Threshold')
    
    # Highlight detected peaks
    ax.scatter(peaks_time, v_signal[peaks], color='red', label='Detected Peaks', zorder=5)

    # Highlight bursts
    for burst in bursts:
        burst_start_time = time[burst[0]]
        burst_end_time = time[burst[-1]]
        burst_indices_time = time[burst]
        ax.scatter(burst_indices_time, v_signal[burst], color='orange', zorder=6, label='Burst' if burst == bursts[0] else "")
        ax.axvspan(burst_start_time, burst_end_time, color='yellow', alpha=0.2, zorder=0)

    ax.set_title("Signal with Detected Peaks and Bursts")
    ax.set_xlabel("Time")
    ax.set_ylabel("Signal Amplitude")
    plt.legend()
    plt.grid()
    # plt.show()



# Reads h5 file with data trials
# Input: h5_file_path: path to the file
#        trials: String with trials separated by whitespace e.g. "1 2 5 7"
#        headers: label to each column in the dataframe
def read_h5File(h5_file_path, trials=None, headers='', column_index=None):
    # Step 2: Read the H5 file
    with h5py.File(h5_file_path, 'r') as f:
        # print("H5 file structure:")
        # f.visit(print)  # Print dataset structure

        all_dataframes = []

        n_trial = 1

        while 1:
            
            #TODO when trials has a value it always need this break
            if n_trial == 30:
                print("Safety break n trial was %d"%n_trial)
                break

            # print(f"Trying Trial {n_trial}...")

            try:
                if trials is not None and n_trial not in trials:
                    n_trial+=1
                    continue

                print(f"Processing Trial {n_trial}...")
                trial = f"Trial{n_trial}"
                struct = f'/{trial}/Synchronous Data/Channel Data'

                dset = f[struct]  # Get trial data
                data = dset[()]  # Get "values"

                # TODO set index in config file
                if column_index is None:
                    columns = range(data.shape[1])  # Replace with specific column indices if needed
                else:
                    columns = column_index

                print(columns)
                # Extract signal and store with trial label
                signal = data[:, columns]
                df_trial = pd.DataFrame(signal, columns=[f"Col{i}" for i in columns])
                df_trial['Trial'] = n_trial  # Add trial identifier
                all_dataframes.append(df_trial)

            except KeyError as e:
                if 'Channel Data' in e.args[0]:
                    print("Skiping Trial %d"%n_trial)
                    n_trial+=1
                    continue
                else:
                    print("No trials left. %d files generated"%n_trial)
                    break
            except Exception as e:
                print(f"Error processing Trial {n_trial}: {e}")
                continue

            n_trial+=1
            
        # Combine all trial data into a single DataFrame
        df = pd.concat(all_dataframes, axis=0, ignore_index=True)

    return df

def save_data(data):
    trial_id = data['Trial']
    column = data['Column_name']
    trial_type = data['Type']
    h5_file_path = data['OPath']

    if len(data['Bursts_Index']) != 0:
        # Save waveforms .pkl
        with open(h5_file_path[:-3] + "_waveform-trial%d-%s-%s.pkl" % (trial_id, column, trial_type), 'wb') as f:
            pickle.dump(data['Waveforms'], f)  # Save waveforms

    # Save as .txt files
    np.savetxt(h5_file_path[:-3] + "_bursts_index-trial%d-%s-%s.txt" % (trial_id, column, trial_type),
            data['Bursts_Index'], fmt="%d", header="Burst Start Index, Burst End Index")
    np.savetxt(h5_file_path[:-3] + "_bursts_time-trial%d-%s-%s.txt" % (trial_id, column, trial_type),
            data['Bursts_Times'], fmt="%.6f", header="Burst Start Time, Burst End Time")

    # Save as .pkl files
    with open(h5_file_path[:-3] + "_bursts_index-trial%d-%s-%s.pkl" % (trial_id, column, trial_type), 'wb') as f:
        pickle.dump(data['Bursts_Index'], f)  # Save burst indices as tuples
    with open(h5_file_path[:-3] + "_bursts_time-trial%d-%s-%s.pkl" % (trial_id, column, trial_type), 'wb') as f:
        pickle.dump(data['Bursts_Times'], f)  # Save burst times as tuples

    # Save peaks and peaks_time as .txt
    np.savetxt(h5_file_path[:-3]+"_spikes_index-trial%d-%s-%s.txt"%(trial_id, column, trial_type), data['Peaks_Index'], fmt="%d")
    np.savetxt(h5_file_path[:-3]+"_spikes_time-trial%d-%s-%s.txt"%(trial_id, column, trial_type), data['Peaks_Times'])
    
    # Save peaks and peaks_time as .pkl
    with open(h5_file_path[:-3] + "_spikes_index-trial%d-%s-%s.pkl" % (trial_id, column, trial_type), 'wb') as f:
        pickle.dump(data['Peaks_Index'].astype(int), f)  # Save peaks as integers
    with open(h5_file_path[:-3] + "_spikes_time-trial%d-%s-%s.pkl" % (trial_id, column, trial_type), 'wb') as f:
        pickle.dump(data['Peaks_Times'], f)  # Save peaks_time (as floats by default)


def main(h5_file_path, config_file_path):
     # Step 1: Read the config file
    config = configparser.ConfigParser()
    config.read(config_file_path)

    save = config['Outcome']['save'].lower() == 'y'
    save_all = config['Outcome']['save_all'].lower() == 'y'
    plot = config['Outcome']['plot'].lower() == 'y'
    print(save, plot)
    
    # Parse recording parameters
    sampling_rate = float(config['Recording']['sampling_rate'])  # in milliseconds
    firing_rate = float(config['Recording']['firing_rate'])  # in Hz

    try:
        percentage_thresholds = config['Spike detection']['threshold']
        percentage_thresholds = tuple([float(thres) for thres in percentage_thresholds.split()])
    except Exception as e:
        print("Error: No percentage threshold for spike detection in Config file or incorrect format")
        print("Message:", e.args)
        exit()

    try:
        columns_to_filter = config['Analysis']['column_to_filter']
        columns_to_filter = tuple([int(col) for col in columns_to_filter.split()])
    except Exception as e:
        print("Warning:", e.args)
        columns_to_filter = ()

    try:
        trials = config['Input']['trials']
        trials = tuple([int(trial) for trial in trials.split()])
    except:
        trials = None

    # Parse burst detection parameters
    min_spike_dist = float(config['Burst detection']['min_spike_dist'])
    max_spike_dist = float(config['Burst detection']['max_spike_dist'])
    min_burst_dist = float(config['Burst detection']['min_burst_dist'])

    # Parse additional input parameters from the config file
    trial_types = config['Input']['type'].strip('"').split()
    column_names = config['Input']['column_names'].strip('"').split()
    try:
        column_index = config['Input']['column_index'].strip().split()
        column_index = [int(x) for x in column_index]
    except:
        column_index = None

    # TODO inlude input check: name, 


    df_signal = read_h5File(h5_file_path, trials, column_index=column_index)
    
    # Update column names in the dataframe
    df_signal.columns = column_names + ['Trial']

    # Map trial numbers to trial types
    trial_type_mapping = dict(zip(trials, trial_types))

    # Add a new column for trial type
    df_signal['Type'] = df_signal['Trial'].map(trial_type_mapping)

    print(df_signal)

    print("Saving Data frame")
    df_signal.to_pickle(h5_file_path[:-3] +'_data.pkl')


    # Provide information about the DataFrame
    print("\nDataFrame Information:")
    print(df_signal.info())
    print("\nDataFrame Head:")
    print(df_signal.head())
    
    extended_data = []
    
    for trial_id in df_signal['Trial'].unique():
        # TODO analyze only "trial_selected"
        trial_data = df_signal[df_signal['Trial'] == trial_id]
        
        n_signals = len(trial_data.columns[:-2])

        fig, ax = plt.subplots(n_signals,figsize=(10, 6))
        for i, column in enumerate(trial_data.columns[:-2]):  # Exclude 'Trial' and 'Type' column
            ax_i = ax[i] if n_signals > 1 else ax


            v_signal = trial_data[column].values # get a neuron signal from a trial

            trial_type = df_signal.loc[df_signal['Trial'] == trial_id, 'Type'].iloc[0]

            if i in columns_to_filter:
                print("Filtering Column %d from Trial %d"%(i, trial_id))
                v_signal = FIR(v_signal, False, 100, 10000)

            peaks, absolute_threshold = get_peaks(v_signal, percentage_thresholds[i], min_distance_ms=min_spike_dist, sampling_rate=sampling_rate)

            # Print detected peaks for debugging
            print(f"Trial {trial_id}, Column {column} detected peaks: {len(peaks)}")

            # save with time dimension            
            time = np.arange(0,v_signal.shape[0],1)*sampling_rate
            peaks_time = time[peaks]
            
            bursts = detect_bursts_from_spikes(peaks, sampling_rate, min_spikes=2, min_spike_dist=min_spike_dist, max_spike_dist=max_spike_dist, min_burst_dist=min_burst_dist)

            # Plot the signal, peaks, and bursts
            plot_signal_with_peaks_and_bursts(ax_i, v_signal, time, peaks, peaks_time, bursts, absolute_threshold)

            # Save burst start and end indices and times
            burst_start_end_indices = [(burst[0], burst[-1]) for burst in bursts]  # Start and end indices of each burst
            burst_start_end_times = [(time[burst[0]], time[burst[-1]]) for burst in bursts]  # Start and end times of each burst
           
            # calculate waveforms
            if len(bursts) != 0:
                # TODO: 1500 hardcoded!!!
                burst_waveforms = [v_signal[burst[0]-1500:1500+burst[1]] for burst in burst_start_end_indices]
                max_length = max(w.shape[0] for w in burst_waveforms)

                burst_waveforms_padded = np.array([np.pad(w, (0, max_length - w.shape[0]), mode='constant') if w.shape[0] < max_length else w[:max_length] for w in burst_waveforms])
            else: # if no burst detected
                burst_waveforms_padded = np.array([])

            # complete dataframe with all information
            extended_data.append({
                'Trial': trial_id,
                'Type': trial_type,
                'Column_id': i,
                'Column_name': i,
                'Time':time,
                'Signal': v_signal,
                'Bursts_Index': burst_start_end_indices,
                'Bursts_Times': burst_start_end_times,
                'Peaks_Index': peaks,
                'Peaks_Times': peaks_time,
                'Waveforms':burst_waveforms_padded,
                'OPath': h5_file_path
            })


            if save_all: # TODO reduce options of saving
                save_data(extended_data[-1])

        if plot:
            plt.tight_layout()
            plt.show()


    if save:
        # Convert extended data to a DataFrame and save it
        df_extended = pd.DataFrame(extended_data)
        # Save the extended DataFrame to a CSV or a pickle file
        extended_path = config_file_path[:-4] + "_extended_data.pkl"
        df_extended.to_pickle(extended_path)

        print(f"Extended DataFrame saved to {extended_path}")
        print(f"Headers: {df_extended.columns}")


# This if ensures that main will not be called when this script is imported by other library
if __name__ == "__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Process an H5 file and a config file.")

    # Define the arguments
    parser.add_argument(
        "h5_file_path",
        type=str,
        help="Path to the H5 file."
    )
    parser.add_argument(
        "config_file_path",
        type=str,
        help="Path to the config (INI) file."
    )

    # Parse the arguments
    args = parser.parse_args()

    # Call main with the parsed arguments
    main(args.h5_file_path, args.config_file_path)


# Example of use:
# python3 detection.py data-test/STG-PD-extra.h5 data-test/STG-PD-extra.ini

