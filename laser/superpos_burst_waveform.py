import pandas as pd
import pickle
import configparser
import numpy as np

# Function to read the config file
def read_config(config_path):
    config = configparser.ConfigParser()
    config.read(config_path)
    
    # Read the necessary config values
    trace_path = config.get('Paths', 'trace_path')
    neuron_type = config.get('Settings', 'neuron_type')
    trial = int(config.get('Settings', 'trial'))
    burst_pkl = config.get('Paths', 'burst_pkl')
    
    return trace_path, neuron_type, trial, burst_pkl

# Function to load the burst index from the pickle file
def load_burst_index(burst_pkl):
    with open(burst_pkl, 'rb') as f:
        burst_index = pickle.load(f)
    return burst_index

# Function to extract the trace for the given neuron, trial, and index
def extract_trace(df, neuron_type, trial, burst_index):
    # Filter dataframe for the specific trial and neuron type
    trace_df = df[(df['Trial'] == trial) & (df['Type'] == neuron_type)]
    
    # Extract the waveform using the burst index
    trace = trace_df.iloc[burst_index]  # burst_index will be an integer or list
    return trace

# Function to save the waveform to a file
def save_waveform(waveform, output_path='waveforms.txt'):
    np.savetxt(output_path, waveform, delimiter=",")
    print(f"Waveform saved to {output_path}")

# Main function to run the process
def main(config_path):
    # Step 1: Read the configuration
    trace_path, neuron_type, trial, burst_pkl = read_config(config_path)
    
    # Step 2: Load the dataframe
    df = pd.read_csv(trace_path)  # Assuming the dataframe is stored as a CSV
    
    # Step 3: Load the burst index
    burst_index = load_burst_index(burst_pkl)
    
    # Step 4: Extract the trace for the specified neuron type, trial, and burst index
    waveform = extract_trace(df, neuron_type, trial, burst_index)
    
    # Step 5: Save the waveform to a file
    save_waveform(waveform)

# Example usage
config_file = '~/Workspace/data/laser/pyloric/11-12-24/Exp1-1450nm-27C.ini'  # The path to the config file
main(config_file)