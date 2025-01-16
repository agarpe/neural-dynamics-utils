import numpy as np
import pandas as pd
import argparse
import configparser
import pickle as pkl
import matplotlib.pyplot as plt


colors= ['cornflowerblue','indianred','yellowgreen']
# --- MAIN FUNCTION ---
def main(config_file_path, data_frame_path):
    """
    Main function to load the DataFrame and process it using configuration.
    Args:
        config_file (str): Path to the configuration file.
        data_file (str): Path to the data file to be loaded.
    """
    # Load config

    config = configparser.ConfigParser()
    config.read(config_file_path)

    # Load dataframe
    df = pd.read_pickle(data_frame_path)

    print(df)

    triplets = [triplet.split() for triplet in config['Superposition']['triplets'].split('|')]
    print(triplets)

    cols = len(triplets)//2
    fig, ax = plt.subplots(2, cols, figsize=(12, 6))
    ax = ax.flatten()
    for i, triplet in enumerate(triplets):
        for j, trial in enumerate(triplet):
            trial = int(trial)
            waveforms = df.loc[df['Trial'] == trial, 'Waveforms'].values[0]
            if j == 1:
                label = df.loc[df['Trial'] == trial, 'Type'].values[0]
            else:
                label = 'control' if i == 0 else 'recovery'

            # Compute average waveform
            try:    
                aligned_waveforms = np.array([w-min(w) for w in waveforms])
                w_mean = np.mean(aligned_waveforms, axis=0)
                # w_mean -= np.max(w_mean)
            except Exception as e:
                print(e.args)
                continue

            # Plot average and traces
            ax[i].plot(aligned_waveforms, color=colors[j], label=label,
                       linewidth=0.01)
            ax[i].plot(w_mean, color=colors[j], label=label)
            if j == 1:
                ax[i].set_title(label)

    plt.tight_layout()
    plt.show()
    exit()

    waveforms_f = config['Superposition']['waveform_files']
    waveforms_f = [w for w in waveforms_f.split()]
    print(waveforms_f)

    title = config['Superposition']['title']

    data_file = config['Superposition']['trace_dataframe']

    # Load DataFrame
    try:
        df = pd.read_pickle(data_file)
        print(f"DataFrame loaded successfully. Shape: {df.shape}")
    except Exception as e:
        print(f"Error loading DataFrame: {e}")
        return

    # Process or use the DataFrame as needed
    print(df.head())

    waveforms = {trial: pkl.load(open(wf, 'rb')) for trial, wf in zip(trials, waveforms_f)}


    for i,w in enumerate(waveforms):
        type_for_trial = df[df['Trial'] == w]['Type'].unique()[0]
        print(type_for_trial)


        try:
            waveform = np.array([w-min(w) for w in waveforms[w]])
            w_mean = np.mean(waveform, axis=0)
            # w_mean -= np.max(w_mean)
        except Exception as e:
            print(e.args)
            continue
        
        
        plt.plot(waveform.T, color=colors[i], label=type_for_trial, linewidth=0.01) 
        plt.plot(w_mean.T, color=colors[i], label=type_for_trial)

    plt.title(title)
    # plt.legend()
    print("Saving at", config_file_path[:-4]+'_'+title+'.png')
    plt.savefig(config_file_path[:-4]+'_'+title+'.png',dpi=200, format='png')

    # plt.show()
    


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
