import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import h5py
import pandas as pd
import configparser
import argparse
import pickle

# file_path = '/media/pablo/External NVME/Recordings/11-12-24/Exp1/17h31m16s_Trial1_Exp1.asc'
# data = np.loadtxt(file_path)
# PD1 = data[:,1]
# Extra = data[:,0]


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

def get_peaks(neuron_signal, percentage_threshold, min_distance):
    # Calculate range of the signal
    signal_range = abs(max(neuron_signal) - min(neuron_signal))

    # Calculate the absolute threshold with respect to the minimum
    absolute_threshold = min(neuron_signal) + percentage_threshold * signal_range

    peaks, _ = signal.find_peaks(neuron_signal, height=absolute_threshold, distance=min_distance, prominence=0.1*signal_range)

    # plt.figure(figsize=(10, 6))
    # plt.plot(neuron_signal, label='neuron signal', color='b')
    # plt.scatter(peaks, neuron_signal[peaks], color='r', marker='x', label='Peaks')
    # plt.hlines(y=absolute_threshold, xmin=0, xmax=len(neuron_signal)-1, color='g', linestyle='--', label='Threshold')
    # # Add labels and title
    # plt.title('Signal with Detected Peaks')
    # plt.xlabel('Time [s]')
    # plt.ylabel('Amplitude')
    # plt.grid(True)
    # plt.legend()

    # # Show the plot
    # plt.tight_layout()
    # plt.show()
    return peaks
# Function to detect bursts
def detect_bursts_from_spikes(spike_indices, min_spikes=3, min_spike_dist=100, max_spike_dist=2000, min_burst_dist=4000):
    """
    Detect bursts from pre-detected spike indices based on spike distances and burst characteristics.

    Parameters:
    - spike_indices: List or array of spike indices (already detected spikes).
    - min_spikes: Minimum number of spikes to consider a burst.
    - min_spike_dist: Minimum distance between spikes within a burst.
    - max_spike_dist: Maximum distance between spikes within a burst.
    - min_burst_dist: Minimum distance between bursts.

    Returns:
    - bursts: List of bursts, each burst is a list of spike indices.
    """
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
def plot_signal_with_peaks_and_bursts(v_signal, time, peaks, peaks_time, bursts):
    plt.figure(figsize=(15, 6))

    # Plot the full signal
    plt.plot(time, v_signal, label='Signal', color='blue', linewidth=1)
    
    # Highlight detected peaks
    plt.scatter(peaks_time, v_signal[peaks], color='red', label='Detected Peaks', zorder=5)

    # Highlight bursts
    for burst in bursts:
        burst_start_time = time[burst[0]]
        burst_end_time = time[burst[-1]]
        burst_indices_time = time[burst]
        plt.scatter(burst_indices_time, v_signal[burst], color='orange', zorder=6, label='Burst' if burst == bursts[0] else "")
        plt.axvspan(burst_start_time, burst_end_time, color='yellow', alpha=0.2, zorder=0)

    plt.title("Signal with Detected Peaks and Bursts")
    plt.xlabel("Time")
    plt.ylabel("Signal Amplitude")
    plt.legend()
    plt.grid()
    plt.show()



# Reads h5 file with data trials
# Input: h5_file_path: path to the file
#        trials: String with trials separated by whitespace e.g. "1 2 5 7"
#        headers: label to each column in the dataframe
def read_h5File(h5_file_path, trials=None, headers=''):
    # Step 2: Read the H5 file
    with h5py.File(h5_file_path, 'r') as f:
        print("H5 file structure:")
        f.visit(print)  # Print dataset structure

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

                # Assuming columns are defined or available from elsewhere
                columns = range(data.shape[1])  # Replace with specific column indices if needed

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



def main(h5_file_path, config_file_path):
     # Step 1: Read the config file
    config = configparser.ConfigParser()
    config.read(config_file_path)

    # Parse recording parameters
    sampling_rate = float(config['Recording']['sampling_rate'])  # in seconds
    max_spike_duration = float(config['Recording']['max_spike_duration'])  # in milliseconds

    # Convert max spike duration to samples
    max_spike_samples = max_spike_duration / sampling_rate

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

    df_signal = read_h5File(h5_file_path, trials)

    # Parse additional input parameters from the config file
    trial_types = config['Input']['type'].strip('"').split()
    column_names = config['Input']['column_names'].strip('"').split()

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

    #TODO decide if loop all or specify which in config
    for trial_id in df_signal['Trial'].unique():
        trial_data = df_signal[df_signal['Trial'] == trial_id]
        
        # TODO: decide if 1 or 2 rows per plot
        # n_signals = len(trial_data.columns[:-1])
        n_signals = 1
        
        for i, column in enumerate(trial_data.columns[:-2]):  # Exclude 'Trial' and 'Type' column
            # fig, ax = plt.subplots(n_signals,figsize=(10, 6))
            # ax_i = ax[i] if n_signals > 1 else ax


            v_signal = trial_data[column].values # get a neuron signal from a trial

            trial_type = df_signal.loc[df_signal['Trial'] == trial_id, 'Type'].iloc[0]

            if i in columns_to_filter:
                print("Filtering Column %d from Trial %d"%(i, trial_id))
                v_signal = FIR(v_signal, False, 100, 10000)

            # TODO: change 100 for config value
            peaks = get_peaks(v_signal, percentage_thresholds[i], 100)

            # # Plot signal and detected peaks
            # ax_i.plot(v_signal, label=f"{column}")
            # ax_i.plot(peaks, v_signal[peaks], "x", label=f"{column} peaks")
            # ax_i.set_title(f'Detected Events for Trial {trial_id}, Column {column}')
            # ax_i.set_xlabel('Index')
            # ax_i.set_ylabel('Signal Value')
            # ax_i.legend()
            # ax_i.grid()

            # Print detected peaks for debugging
            print(f"Trial {trial_id}, Column {column} detected peaks: {len(peaks)}")

            # save with time dimension            
            time = np.arange(0,v_signal.shape[0],1)*sampling_rate
            peaks_time = time[peaks]

            # plt.plot(time, v_signal, label=f"{column}")
            # plt.plot(peaks_time, np.zeros(peaks_time.shape[0]),'x', label=f"{column}")
            # plt.show()
            
            bursts = detect_bursts_from_spikes(peaks, min_spikes=3, min_spike_dist=100, max_spike_dist=2000, min_burst_dist=4000)

            # Plot the signal, peaks, and bursts
            plot_signal_with_peaks_and_bursts(v_signal, time, peaks, peaks_time, bursts)


            # Save burst start and end indices and times
            burst_start_end_indices = [(burst[0], burst[-1]) for burst in bursts]  # Start and end indices of each burst
            burst_start_end_times = [(time[burst[0]], time[burst[-1]]) for burst in bursts]  # Start and end times of each burst
            
            if len(bursts) != 0:
                burst_waveforms = [v_signal[burst[0]-2000:2000+burst[1]] for burst in burst_start_end_indices]
                min_length = min(w.shape[0] for w in burst_waveforms)
                trimmed_waveforms = np.array([w[:min_length] for w in burst_waveforms])
                
                # Save peaks and peaks_time as .pkl
                with open(h5_file_path[:-3] + "_waveform-trial%d-%s-%s.pkl" % (trial_id, column, trial_type), 'wb') as f:
                    pickle.dump(trimmed_waveforms, f)  # Save waveforms

            # Save as .txt files
            np.savetxt(h5_file_path[:-3] + "_bursts_index-trial%d-%s-%s.txt" % (trial_id, column, trial_type),
                    burst_start_end_indices, fmt="%d", header="Burst Start Index, Burst End Index")
            np.savetxt(h5_file_path[:-3] + "_bursts_time-trial%d-%s-%s.txt" % (trial_id, column, trial_type),
                    burst_start_end_times, fmt="%.6f", header="Burst Start Time, Burst End Time")

            # Save as .pkl files
            with open(h5_file_path[:-3] + "_bursts_index-trial%d-%s-%s.pkl" % (trial_id, column, trial_type), 'wb') as f:
                pickle.dump(burst_start_end_indices, f)  # Save burst indices as tuples
            with open(h5_file_path[:-3] + "_bursts_time-trial%d-%s-%s.pkl" % (trial_id, column, trial_type), 'wb') as f:
                pickle.dump(burst_start_end_times, f)  # Save burst times as tuples

            np.savetxt(h5_file_path[:-3]+"_spikes_index-trial%d-%s-%s.txt"%(trial_id, column, trial_type), peaks, fmt="%d")
            np.savetxt(h5_file_path[:-3]+"_spikes_time-trial%d-%s-%s.txt"%(trial_id, column, trial_type), peaks_time)
            
            # Save peaks and peaks_time as .pkl
            with open(h5_file_path[:-3] + "_spikes_index-trial%d-%s-%s.pkl" % (trial_id, column, trial_type), 'wb') as f:
                pickle.dump(peaks.astype(int), f)  # Save peaks as integers
            with open(h5_file_path[:-3] + "_spikes_time-trial%d-%s-%s.pkl" % (trial_id, column, trial_type), 'wb') as f:
                pickle.dump(peaks_time, f)  # Save peaks_time (as floats by default)



        # plt.show()

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




# OLD code:
# # Example usage:
# spike_indices = np.array([50, 51, 52, 120, 121, 122, 300, 301, 320, 330, 350])  # Pre-detected spike indices

# # Detect bursts with parameters:
# bursts = detect_bursts_from_spikes(spike_indices, min_spikes=3, min_spike_dist=10, max_spike_dist=50, min_burst_dist=100)

# # Print detected bursts
# print("Detected Bursts:")
# for i, burst in enumerate(bursts):
#     print(f"Burst {i + 1}: {burst}")



# filtered_PD1 = FIR(PD1, False, 100, 10000)
# PD1_spikes = get_peaks(filtered_PD1, 0.001, 100)

# LP_spikes = get_peaks(Extra, 0.08, 100)

# print(detect_bursts_from_spikes(PD1_spikes))

