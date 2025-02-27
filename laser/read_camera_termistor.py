import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import argparse

def parse_standard_file(filepath):
    """ Parses the first file format (semicolon-separated, HH:MM:SS,fff time format). """
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
    
    """ Parses the new file format (comma-separated, HH:MM:SS.SSS time format). """
    #df = pd.read_csv(filepath, sep=',', skiprows=1, usecols=[0, 1], names=['Time', 'Temperature'], encoding='utf-8', skip_blank_lines=True)

    # print(df)
    # # Convert 'Time' column to datetime format
    # df['Time'] = pd.to_datetime('1900-01-01 ' + df['Time'], format='%Y-%m-%d %H:%M:%S.%f')
    #
    # # Convert 'Temperature' column to float, handling empty values
    # df['Temperature'] = pd.to_numeric(df['Temperature'], errors='coerce')  # Converts invalid values to NaN

    return df

def plot_data(ax, time, temperature, title):
    """ Plots Time vs Temperature """
    ax.plot(time, temperature, marker='o', linestyle='-', markersize = 2, label= 'Temperature Over Time'+title)
    ax.set_xlabel('Time')
    ax.set_ylabel('Temperature (°C)')
    
    plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
    # plt.grid()

# Create the argument parser
parser = argparse.ArgumentParser(description="Parses a .dat from OPI camera to dataframe and .csv from thermistor.")

# Define the arguments
parser.add_argument(
    "dat_file_path",
    type=str,
    help="Path to the .dat file."
)
# Define the arguments
parser.add_argument(
    "csv_file_path",
    type=str,
    help="Path to the .csv file."
)
# Parse the arguments
args = parser.parse_args()


# Example usage:
standard_filepath = args.dat_file_path  # Replace with actual file path
comma_filepath = args.csv_file_path  # Replace with actual file path

df1 = parse_standard_file(standard_filepath)
df2 = parse_comma_file(comma_filepath)
df2 = df2.iloc[::10]

# Determine the maximum length of the 'Temperature' column
shared_time_length = max(df1['Temperature'].shape[0], df2['Termistor'].shape[0])
# Create the shared_time array from 0 to N with step 1
shared_time = np.arange(0, shared_time_length)



#TODO: Función para estos dos ifs
# If df1 has fewer rows, pad with NaNs
if df1['Temperature'].shape[0] < shared_time_length:
    padding_len = shared_time_length - df1['Temperature'].shape[0]
    print(padding_len)
    
    
    # Pad the values of 'Termistor' with NaNs to match the shared_time_length
    padded_data = np.pad(df1['Temperature'].values, (padding_len, 0), constant_values=np.nan)

    # Reset the index to match the shared_time_length
    df1 = df1.reset_index(drop=True)

    # Ensure the DataFrame has the correct number of rows
    df1 = df1.reindex(range(shared_time_length))
    
    df1['Temperature'] = padded_data

# If df2 has fewer rows, pad with NaNs
if df2['Termistor'].shape[0] < shared_time_length:
    padding_len = shared_time_length - df2['Termistor'].shape[0]
    print("Padding length:", padding_len)

    # Pad the values of 'Termistor' with NaNs to match the shared_time_length
    padded_data = np.pad(df2['Termistor'].values, (padding_len, 0), constant_values=np.nan)

    # Reset the index to match the shared_time_length
    df2 = df2.reset_index(drop=True)

    # Ensure the DataFrame has the correct number of rows
    df2 = df2.reindex(range(shared_time_length))
    
    df2['Termistor'] = padded_data

# Now both df1 and df2 have the same number of rows

###PLOT###
fig, ax = plt.subplots(1, 1, figsize=(10, 8))


plot_data(ax, shared_time,df1['Temperature'], ' camera')  # Plot first dataset
plot_data(ax, shared_time,df2['Termistor'], ' Termistor')  # Plot second dataset
# plot_data(ax, df2['Time'], df2['Termistor'], 'Termistor')  # Plot second dataset
#plot_data(df2['Time'],df2['TermistorWater'], 'TermistorWater')
plt.legend()

plt.show()