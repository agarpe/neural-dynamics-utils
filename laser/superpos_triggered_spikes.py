# read extended_data pkl

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
    
    # TODO get from trial N the waveforms

    # TODO get from trial N the stimulus

    # TODO get from pkl the column "spike_times"

    # TODO get difference between input and 

    # Waveform analysis
    time_step = df['Time'].iloc[0]  # e.g., 20000 Hz → 1/20000 s
    dt = time_step[1]

    # # Plot waveforms for triplets
    plot_triplet_waveforms(df, triplets, column_id, title, save_prefix, dt, vscale=1000)
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
