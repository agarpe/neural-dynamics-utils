import numpy as np
import pandas as pd
import argparse
import configparser
import pickle as pkl
import matplotlib.pyplot as plt

def plot_waveforms():


colors= ['cornflowerblue','crimson','yellowgreen']
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
    column_id = int(config['Superposition']['column_id'])
    print(triplets)

    cols = len(triplets)//2
    fig, ax = plt.subplots(2, cols, figsize=(12, 8))
    ax = ax.flatten()

    fig_mean, ax_mean = plt.subplots(2, cols, figsize=(12, 8))
    ax_mean = ax_mean.flatten()

    title = data_frame_path[data_frame_path.rfind('/')+1:
                            data_frame_path.rfind('_extended_data.pkl')]
    for i, triplet in enumerate(triplets):
        for j, trial in enumerate(triplet):
            trial = int(trial)
            waveforms = df.loc[(df['Trial'] == trial) & 
                               (df['Column_id'] == column_id),
                               'Waveforms'].values[0]
            if j == 1:
                label = df.loc[df['Trial'] == trial, 'Type'].values[0]
            else:
                label = 'control' if i == 0 else 'recovery'

            # Compute average waveform
            try:
                # TODO align to first spike
                aligned_waveforms = np.array([w-min(w) for w in waveforms])
                w_mean = np.mean(aligned_waveforms, axis=0)
                # w_mean -= np.max(w_mean)
            except Exception as e:
                print(e.args)
                continue

            # Plot average and traces
            ax[i].plot(aligned_waveforms.T, color=colors[j], label=label,
                       linewidth=0.01)
            ax[i].plot(w_mean.T, color=colors[j], label=label)
            ax_mean[i].plot(w_mean.T, color=colors[j], label=label)
            if j == 1:
                ax[i].set_title(label)
                ax_mean[i].set_title(label)

    fig.suptitle(title)    
    fig_mean.suptitle(title)    

    fig.tight_layout()
    fig_mean.tight_layout()
    
    print(title)

    print("Saving at", config_file_path[:-4]+'_'+title+'.png')
    fig.savefig(config_file_path[:-4]+'_'+title+'.png', dpi=200, format='png')
    fig_mean.savefig(config_file_path[:-4]+'_'+title+'_average.pdf', dpi=200, format='pdf')

    plt.show()
    


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
