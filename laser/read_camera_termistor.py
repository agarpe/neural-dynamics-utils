import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import argparse

def parse_standard_file(filepath):
    """ Parses the first file format (semicolon-separated, HH:MM:SS,fff time format). """
    print(filepath)
    with open(filepath, 'r', encoding='utf-8', errors='replace') as file:
        lines = file.readlines()

    # Find the first line that contains actual data
    for i, line in enumerate(lines):
        if ',' in line and line[0].isdigit():  # Ensure it starts with a digit
            data_start = i
            break

    # Read the file
    df = pd.read_csv(filepath, sep=',', skiprows=data_start, names=['Time', 'Temperature'], encoding='utf-8', index_col=False, skip_blank_lines=True)

    # Remove unwanted rows
    df = df[~df['Time'].str.contains("---|End of File", na=False)]

    # Convert 'Time' column to datetime format
    df['Time'] = pd.to_datetime('1900-01-01 ' + df['Time'], format='%Y-%m-%d %H:%M:%S.%f')

    # Convert 'Temperature' column to float
    df['Temperature'] = df['Temperature'].astype(float)
    print(df)
    return df

def parse_comma_file(filepath):
        # First, check how many columns are in the CSV
    with open(filepath, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()  # Read just the first line
        num_columns = len(first_line.split(','))  # Count how many columns there are

    # Adjust the `usecols` and `names` based on the number of columns
    if num_columns == 2:
        df = pd.read_csv(filepath, sep=',', skiprows=1, usecols=[0, 1], names=['Time', 'Termistor'], encoding='utf-8', skip_blank_lines=True)
    elif num_columns == 3:
        df = pd.read_csv(filepath, sep=',', skiprows=1, usecols=[0, 1, 2], names=['Time', 'Termistor', 'TermistorWater'], encoding='utf-8', skip_blank_lines=True)
    else:
        raise ValueError("Unexpected number of columns in CSV file")

    # Display the DataFrame
    print(df)

    return df

def plot_data(ax, time, temperature, title):
    """ Plots Time vs Temperature """
    ax.plot(time, temperature, marker='o', linestyle='-', markersize = 2, label= 'Temperature Over Time'+title)
    ax.set_xlabel('Time')
    ax.set_ylabel('Temperature (°C)')
    
    plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
    # plt.grid()


# Padding function
def pad_dataframe(df, column_name, target_length):
    """ Pads the specified column with NaNs if needed. """
    if df[column_name].shape[0] < target_length:
        padding_len = target_length - df[column_name].shape[0]
        padded_data = np.pad(df[column_name].values, (0, padding_len), constant_values=np.nan)
        df = df.reset_index(drop=True).reindex(range(target_length))
        df[column_name] = padded_data
    return df

def read_key_times(file_path):
    """ Reads key times from a file and converts them to indices. """
    with open(file_path, 'r') as file:
        times = file.read().splitlines()

    key_indices = []
    for t in times:
        mm, ss = map(int, t.split(':'))
        total_seconds = mm * 60 + ss  # Convert MM:SS to total seconds
        key_indices.append(total_seconds)

    return key_indices

# Create the argument parser
parser = argparse.ArgumentParser(description="Parses a .dat from OPI camera to dataframe and .csv from thermistor.")

# Define the arguments
parser.add_argument("dat_file_path", type=str, help="Path to the .dat file.")
parser.add_argument("csv_file_path", type=str, help="Path to the .csv file.")
parser.add_argument("key_times_file", type=str, help="Path to the key times file.")  # New argument for key times
args = parser.parse_args()

standard_filepath = args.dat_file_path  # Replace with actual file path to ignore arguments
comma_filepath = args.csv_file_path  # Replace with actual file path to ignore arguments
key_times_filepath = args.key_times_file

df1 = parse_standard_file(standard_filepath)
# df3 = parse_standard_file("/home/agarpe/Workspace/data/laser/thermal-camera-precission/test4_cameraonthetop_mean.dat")
df2 = parse_comma_file(comma_filepath)
df2 = df2.iloc[::10]

# Determine the maximum length of the 'Temperature' column
shared_time_length = max(df1['Temperature'].shape[0], df2['Termistor'].shape[0])
# Create the shared_time array from 0 to N with step 1
shared_time = np.arange(0, shared_time_length)

df1 = pad_dataframe(df1, 'Temperature', shared_time_length)
df2 = pad_dataframe(df2, 'Termistor', shared_time_length)
try:
    df2 = pad_dataframe(df2, 'TermistorWater', shared_time_length)
except:
    pass
try:
    df3 = pad_dataframe(df3, 'Temperature', shared_time_length)
except:
    pass

# Read key times and get indices
key_indices = read_key_times(key_times_filepath)

# Now both df1 and df2 have the same number of rows

###PLOT###
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

plot_data(ax, shared_time, df1['Temperature'], ' camera')  # Plot first dataset
plot_data(ax, shared_time, df2['Termistor'], ' Termistor')  # Plot second dataset
try:
    plot_data(ax, shared_time, df2['TermistorWater'], ' Termistor Water')  # Plot second dataset
except:
    pass

try:
    plot_data(ax, shared_time, df3['Temperature'], ' camera mean')  # Plot first dataset
except:
    pass


# Plot vertical lines at key times
for key_idx in key_indices:
    if key_idx < shared_time_length:
        ax.axvline(x=key_idx, color='r', linestyle='--', linewidth=0.5, label="Key Time" if key_idx == key_indices[0] else "")

plt.legend()


# plt.xlim(800,1600)


plt.tight_layout()
save_path = standard_filepath[:standard_filepath.rfind('test')+5]+"graph.png"
plt.savefig(save_path, dpi=200, format='png')

plt.show()