import pandas as pd
import json

# --- CONFIG LOADER ---
def load_config(file_path):
    """
    Load configuration from a JSON file.
    Args:
        file_path (str): Path to the config file.
    Returns:
        dict: Configuration parameters.
    """
    with open(file_path, 'r') as f:
        config = json.load(f)
    return config

# --- MAIN FUNCTION ---
def main(config_file, data_file):
    """
    Main function to load the DataFrame and process it using configuration.
    Args:
        config_file (str): Path to the configuration file.
        data_file (str): Path to the data file to be loaded.
    """
    # Load config
    config = load_config(config_file)
    trials = config.get('trials', None)
    
    if trials is None:
        print("Error: 'trials' parameter not found in the configuration.")
        return
    
    print(f"Trials parameter: {trials}")
    
    # Load DataFrame
    try:
        df = pd.read_csv(data_file)
        print(f"DataFrame loaded successfully. Shape: {df.shape}")
    except Exception as e:
        print(f"Error loading DataFrame: {e}")
        return

    # Process or use the DataFrame as needed
    print(df.head())

if __name__ == "__main__":
    # Example usage
    main(config_file="config.json", data_file="data.csv")
