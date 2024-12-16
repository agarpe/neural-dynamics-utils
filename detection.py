import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import h5py
import pandas as pd
import configparser
import numpy as np

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

def get_peaks(neuron_signal, threshold, min_distance):

	peaks, _ = signal.find_peaks(neuron_signal, height=threshold, distance=min_distance)

	# plt.figure(figsize=(10, 6))
	# plt.plot(neuron_signal, label='neuron signal', color='b')
	# plt.scatter(peaks, neuron_signal[peaks], color='r', marker='x', label='Peaks')

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
            current_burst.append(spike_indices[i - 1])
        
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

    # # Display config content for debugging
    # print("Config file contents:")
    # for section in config.sections():
    #     print(f"[{section}]")
    #     for key, value in config[section].items():
    #         print(f"{key} = {value}")


    # Parse recording parameters
    sampling_rate = float(config['Recording']['sampling_rate'])  # in seconds
    max_spike_duration = float(config['Recording']['max_spike_duration'])  # in milliseconds

    # Convert max spike duration to samples
    max_spike_samples = max_spike_duration / (sampling_rate)

    try:
        columns_to_filter = config['Analysis']['columns_to_filter']
        columns_to_filter = tuple([int(col) for col in columns_to_filter.split()])
    except:
        columns_to_filter =()
    try:
        trials = config['Recording']['trials']
        trials = tuple([int(trial) for trial in trials.split()])
    except:
        trials = None

    df_signal = read_h5File(h5_file_path, trials)

    # Step 3: Provide information about the DataFrame
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
        
        for i, column in enumerate(trial_data.columns[:-1]):  # Exclude 'Trial' column
            # if i == 0:
            #     continue
            fig, ax = plt.subplots(n_signals,figsize=(10, 6))
            ax_i = ax[i] if n_signals > 1 else ax

            v_signal = trial_data[column].values # get a neuron signal from a trial

            # if i in columns_to_filter:
            #     print("Filtering Column %d from Trial %d"%(i, trial_id))
            #     v_signal = FIR(v_signal, False, 100, 10000)

            # peaks = get_peaks(v_signal, 0.001, 100)
            peaks = get_peaks(v_signal, 0.08, 100)

            # print(v_signal.max(), v_signal.min(), v_signal.mean())
            # input()


            # LP_spikes = get_peaks(Extra, 0.08, 100)

            # print(detect_bursts_from_spikes(PD1_spikes))


            # Plot signal and detected peaks
            ax_i.plot(v_signal, label=f"{column}")
            ax_i.plot(peaks, v_signal[peaks], "x", label=f"{column} peaks")
            ax_i.set_title(f'Detected Events for Trial {trial_id}, Column {column}')
            ax_i.set_xlabel('Index')
            ax_i.set_ylabel('Signal Value')
            ax_i.legend()
            ax_i.grid()



            # Print detected peaks for debugging
            print(f"Trial {trial_id}, Column {column} detected peaks:")

        plt.show()



# Replace with the paths to your H5 file and config file
h5_file_path = 'data-test/STG-PD-extra.h5'
config_file_path = 'data-test/STG-PD-extra.ini'


main(h5_file_path, config_file_path)


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

