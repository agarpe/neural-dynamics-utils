import charact_utils as utils 
import argparse 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import scipy.signal as signal

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

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=True, help="Path to the file to analyze")
ap.add_argument("-e", "--extension", required=True, help="File extension")
ap.add_argument("-c", "--column", required=False,default=0, help="Column")
ap.add_argument("-sh", "--show", required=False,default='y', help="Show plot")
ap.add_argument("-sv", "--save", required=False,default='y', help="Save events")

args = vars(ap.parse_args())

path = args['path']

show= True if args['show']=='y' else False 
save= True if args['save']=='y' else False 

denoise =True


col = int(args['column'])
ext = args['extension']

	
try:
	df = pd.read_csv(path, delimiter = " ",skiprows=1,header=None)
except:
	print("Error: file not found",path)
	exit()

print("Denoising from ",path)

data = df[col]

# # beta: denoise signal
# if denoise:
# 	# d_data = signal.symiirorder1(data,10,0.7)
# 	# d_data = signal.wiener(data,10,0.7)
# 	d_data = signal.savgol_filter(data, 21, 3)

# plt.plot(data)
# plt.plot(d_data)
# plt.show()


if denoise:
	Wn = 0.008
	N=5
	b, a = signal.butter(N, Wn, 'low')
	d_data = signal.filtfilt(b, a, data)

plt.plot(data[:50000])
plt.plot(d_data[:50000])
plt.show()

#TODO: save new signal