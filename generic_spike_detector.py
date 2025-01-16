import h5py
import pandas as pd
import configparser
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

def main(h5_file_path, config_file_path):
    # Step 1: Read the config file
    config = configparser.ConfigParser()
    config.read(config_file_path)

    # Parse recording parameters
    sampling_rate = float(config['Recording']['sampling_rate'])  # in seconds
    max_spike_duration = float(config['Recording']['max_spike_duration'])  # in milliseconds

    # Display config content for debugging
    print("Config file contents:")
    for section in config.sections():
        print(f"[{section}]")
        for key, value in config[section].items():
            print(f"{key} = {value}")

    # Convert max spike duration to samples
    max_spike_samples = max_spike_duration / (sampling_rate)

    # Step 2: Read the H5 file
    with h5py.File(h5_file_path, 'r') as f:
        print("H5 file structure:")
        f.visit(print)  # Print dataset structure

        # Extract relevant data from H5 file
        trials = [1, 2, 3]  # Example trial indices, replace with your logic
        all_dataframes = []

        for n_trial in trials:
            print(f"Processing Trial {n_trial}...")
            trial = f"Trial{n_trial}"
            struct = f'/{trial}/Synchronous Data/Channel Data'

            try:
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
                print(f"Skipping Trial {n_trial} due to missing data.")
                continue
            except Exception as e:
                print(f"Error processing Trial {n_trial}: {e}")
                continue

        # Combine all trial data into a single DataFrame
        df = pd.concat(all_dataframes, axis=0, ignore_index=True)

    # Step 3: Provide information about the DataFrame
    print("\nDataFrame Information:")
    print(df.info())
    print("\nDataFrame Head:")
    print(df.head())

    # Step 4: Detect and plot events for each trial and column
    print("\nDetecting events per trial and column...")
    for trial_id in df['Trial'].unique():
        trial_data = df[df['Trial'] == trial_id]

        for column in trial_data.columns[:-1]:  # Exclude 'Trial' column
            signal = trial_data[column].values

            v_signal += np.min(v_signal)

            # get the range of the voltage signal 
            signal_range = np.max(v_signal) - np.min(v_signal)
            prominence_percentage = np.max(v_signal)*0.9
            print(prominence_percentage)

            # Detect peaks
            peaks, properties = signal.find_peaks(v_signal, width=(0, max_spike_samples), prominence=prominence_percentage)

            # Plot signal and detected peaks
            plt.figure(figsize=(10, 6))
            plt.plot(signal, label=f"{column}")
            plt.plot(peaks, signal[peaks], "x", label=f"{column} peaks")

            plt.title(f'Detected Events for Trial {trial_id}, Column {column}')
            plt.xlabel('Index')
            plt.ylabel('Signal Value')
            plt.legend()
            plt.grid()
            plt.show()

            # Print detected peaks for debugging
            print(f"Trial {trial_id}, Column {column} detected peaks:")
            print(f"  {len(peaks)} peaks at indices {peaks}")


    # # Step 4: Plot the signals for each trial
    # print("\nPlotting signals per trial...")
    # for trial_id in df['Trial'].unique():
    #     plt.figure(figsize=(10, 6))
    #     trial_data = df[df['Trial'] == trial_id]
    #     for column in trial_data.columns[:-1]:  # Exclude 'Trial' column
    #         plt.plot(trial_data.index, trial_data[column], label=f"{column}")

    #     plt.title(f'Signals from Trial {trial_id}')
    #     plt.xlabel('Index')
    #     plt.ylabel('Signal Value')
    #     plt.legend()
    #     plt.grid()
    #     plt.show()





# Replace with the paths to your H5 file and config file
h5_file_path = '../data/laser/pyloric/09-12-24/Registro electrofisiológico/Exp1.h5'
config_file_path = '../data/laser/pyloric/09-12-24/Registro electrofisiológico/Exp1.ini'

main(h5_file_path, config_file_path)