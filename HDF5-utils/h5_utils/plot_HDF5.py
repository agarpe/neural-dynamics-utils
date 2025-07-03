#################################################################################################
### Plot HDF5 Trials - Each channel in separate subplot row, shared x-axis
#################################################################################################

import h5py
import matplotlib.pyplot as plt
import numpy as np
import argparse

def get_date(f, trial):
	struct = f'/{trial}/Date'
	data = f[struct][()]
	data = data.decode('UTF-8')
	data = data[data.rfind('T')+1:]
	h, m, s = data.split(':')
	return f"{h}h{m}m{s}s"

# Argument parsing
ap = argparse.ArgumentParser(description="Plot trials from an HDF5 file with one subplot per channel.")

ap.add_argument("-p", "--path", required=True, help="Path to the HDF5 file")
ap.add_argument("-c", "--cols", required=False, help="Space-separated list of column indices (e.g., '0 1 2')")
ap.add_argument("-t", "--trials", required=False, help="Comma-separated list of trial numbers (e.g., '1,2,5')")

args = ap.parse_args()

filename = args.path

# Optional columns
columns = None
if args.cols:
	columns = tuple(int(i) for i in args.cols.split())

# Optional trial filter
selected_trials = None
if args.trials:
	selected_trials = set("Trial" + str(i.strip()) for i in args.trials.split(","))

# Open file
try:
	f = h5py.File(filename, 'r')
except Exception as e:
	print("Error opening file:", e)
	exit(1)

i = 1
while True:
	trial = f"Trial{i}"

	if selected_trials and trial not in selected_trials:
		i += 1
		continue

	struct = f'/{trial}/Synchronous Data/Channel Data'
	try:
		data = f[struct][()]
	except KeyError:
		if not selected_trials:
			break
		else:
			i += 1
			continue

	# Use all columns if not specified
	if columns is None:
		columns_to_plot = range(data.shape[1])
	else:
		columns_to_plot = columns

	signal = data[:, columns_to_plot]
	num_channels = len(columns_to_plot)

	# Subplots
	fig, axs = plt.subplots(num_channels, 1, sharex=True, figsize=(10, 2.5 * num_channels))
	if num_channels == 1:
		axs = [axs]  # make iterable if only one

	for j, ch in enumerate(columns_to_plot):
		axs[j].plot(signal[:, j], label=f'ch{ch}')
		axs[j].set_ylabel(f'ch{ch}')
		axs[j].legend(loc='upper right')
		axs[j].grid(True)

	axs[-1].set_xlabel("Sample")
	fig.suptitle(f"{trial} - {get_date(f, trial)}")

	plt.tight_layout()
	plt.subplots_adjust(top=0.9)  # make room for title
	plt.show()

	i += 1

	if selected_trials and trial in selected_trials:
		selected_trials.remove(trial)
		if not selected_trials:
			break
