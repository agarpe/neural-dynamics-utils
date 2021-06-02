import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy import signal
import scipy.fftpack

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def plot_fft(data):
	# Number of samplepoints
	N = len(data)
	# sample spacing
	T = 1.0 / 800.0
	yf = scipy.fftpack.fft(data)
	xf = np.linspace(0.0, 1.0/(2.0*T), N/2)

	fig, ax = plt.subplots()
	ax.plot(xf, 2.0/N * np.abs(yf[:N//2]))
	plt.show()


def filt_extracellular(v):
	minv = np.min(v[:15000])
	maxv = np.max(v[:15000])
	rangev = maxv - minv
	th_up = 0.9
	th_lo = 0.1
	mean_pts = 3

	v_filt = []
	for k in range(len(v)):
		if (k > 100 and k < len(v)-100 and v[k] < (minv + rangev * th_up) and v[k] > (minv + rangev * th_lo)):
			v_filt.append(np.mean(v[k-mean_pts:k+mean_pts]))
		else:
			v_filt.append(v[k])

		if (k % 15000 == 0 and k > 15000 and k < len(v)-15000):
			minv = np.min(v[k-15000:k+15000])
			maxv = np.max(v[k-15000:k+15000])
			rangev = maxv - minv

	return v_filt


def filt_intracellular(v):
	minv = np.min(v[:15000])
	maxv = np.max(v[:15000])
	rangev = maxv - minv
	th_up = 0.7
	mean_pts_up = 10
	mean_pts_lo = 100

	v_filt = []
	for k in range(len(v)):
		if (k > 100 and k < len(v)-100 and v[k] < (minv + rangev * th_up)):
			v_filt.append(np.mean(v[k-mean_pts_lo:k+mean_pts_lo]))
		elif (k > 100 and k < len(v)-100 and v[k] >= (minv + rangev * th_up)):
			v_filt.append(np.mean(v[k-mean_pts_up:k+mean_pts_up]))
		else:
			v_filt.append(v[k])

		if (k % 15000 == 0 and k > 15000 and k < len(v)-15000):
			minv = np.min(v[k-15000:k+15000])
			maxv = np.max(v[k-15000:k+15000])
			rangev = maxv - minv

	return v_filt

filename = "2019y7m5d/toma_inv_6.txt"
#filename = "signal_model_robot.txt"
dataset = pd.read_csv(filename, delimiter=' ', header=0)
data = dataset.values
i = [x / 10000 for x in data[:,0]]
c = data[:,1]
v_pd = data[:,2]
v_lp = data[:,3]
e_pd = data[:,4]
e_lp = data[:,5]
period = [x / 1000 for x in data[:,6]]

v_lp[::2] = -v_lp[::2] # reconstruir el extracelular

v_lp_filt = filt_extracellular(v_lp)

plt.plot(v_lp)
plt.plot(v_lp_filt)
plt.show()

v_pd_filt = filt_intracellular(v_pd)
plt.plot(v_pd)
plt.plot(v_pd_filt)
plt.show()


#v_pd = [x + 3 for x in v_pd]

interval_lppd = []
last_lp = 0
interval = 0
for k in range(len(i)):
	if (e_pd[k] == 1):
		interval = k - last_lp

	if (e_lp[k] == 1):
		last_lp = k

	interval_lppd.append(interval)

interval_lppd = [x / 10 / 1000 for x in interval_lppd]
'''
slope_interval, intercept_interval, r_interval, pvalue_interval, std_error_interval = stats.linregress(period, interval_lppd)
r2_interval = r_interval*r_interval


plt.scatter(period, interval_lppd, c=np.linspace(0, len(interval_lppd), len(period)), cmap=plt.get_cmap("Blues"))
plt.colorbar()
plt.plot(period, intercept_interval+(slope_interval*np.asarray(period)), alpha=0.5, color='blue', label="R2 LPPD=%f"%r2_interval)

plt.legend()

plt.xlabel('Period (ms)')
plt.ylabel('Amplitude')

plt.show()
'''


# plot it
f, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 2]}, sharex = True)
a0.plot(i, v_lp_filt, "lightseagreen")
a0.set_ylabel("Voltage (mV)")
a0.set_xlim(left=0, right=i[-1])

a1.plot(i, v_pd_filt, "darkolivegreen")
a1.set_ylabel("Voltage (mV)")
a1.set_xlim(left=0, right=i[-1])


#f.xlabel("Time (s)")
f.tight_layout()
f.show()
plt.show()


# plot it
f, (a0, a1, a2, a3) = plt.subplots(4, 1, gridspec_kw={'height_ratios': [2, 2, 1, 5]}, sharex = True)
a0.plot(i, v_lp, "lightseagreen")
a0.set_ylabel("Voltage (mV)")
a0.set_xlim(left=2, right=i[-1])

a1.plot(i, v_pd, "darkolivegreen")
a1.set_ylabel("Voltage (mV)")
a1.set_xlim(left=2, right=i[-1])


a2.plot(i, c, "orange")
a2.set_ylabel("Injected current (nA)")
a2.set_xlim(left=2, right=i[-1])


a3.plot(i, period)
a3.plot(i, interval_lppd)
a3.set_ylabel("Burst duration (s)")
#a3.set_ylim(bottom=0.55, top=0.85)
a3.set_xlim(left=2, right=i[-1])


#f.xlabel("Time (s)")
f.tight_layout()
f.show()
plt.show()

