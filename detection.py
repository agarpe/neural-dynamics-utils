import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt


file_path = '/media/pablo/External NVME/Recordings/11-12-24/Exp1/17h31m16s_Trial1_Exp1.asc'
data = np.loadtxt(file_path)
PD1 = data[:,1]
Extra = data[:,0]


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

	plt.figure(figsize=(10, 6))
	plt.plot(neuron_signal, label='neuron signal', color='b')
	plt.scatter(peaks, neuron_signal[peaks], color='r', marker='x', label='Peaks')

	# Add labels and title
	plt.title('Signal with Detected Peaks')
	plt.xlabel('Time [s]')
	plt.ylabel('Amplitude')
	plt.grid(True)
	plt.legend()

	# Show the plot
	plt.tight_layout()
	plt.show()
	return peaks

import numpy as np

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


# Example usage:
spike_indices = np.array([50, 51, 52, 120, 121, 122, 300, 301, 320, 330, 350])  # Pre-detected spike indices

# Detect bursts with parameters:
bursts = detect_bursts_from_spikes(spike_indices, min_spikes=3, min_spike_dist=10, max_spike_dist=50, min_burst_dist=100)

# Print detected bursts
print("Detected Bursts:")
for i, burst in enumerate(bursts):
    print(f"Burst {i + 1}: {burst}")



filtered_PD1 = FIR(PD1, False, 100, 10000)
PD1_spikes = get_peaks(filtered_PD1, 0.001, 100)

LP_spikes = get_peaks(Extra, 0.08, 100)

print(detect_bursts_from_spikes(PD1_spikes))


