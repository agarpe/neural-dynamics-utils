import charact_utils as utils 
import argparse 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import scipy.signal as signal
import os

# WARNING CHANGES AFFECTING ELECTRICAL AUTOM
##### scale default 1

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=True, help="Path to the file to analyze")
ap.add_argument("-e", "--extension", required=True, help="File extension")
ap.add_argument("-c", "--column", required=False,default=0, help="Column")
ap.add_argument("-sc", "--scale", required=False, default=1, help="Scale for mV. Signal*scale")
ap.add_argument("-tol", "--tol", required=False, default=0.5, help="threshold detection tolerance")
ap.add_argument("-dt", "--time_step", required=False, default=0.1, help="Time step")
ap.add_argument("-ws_l", "--window_size_l", required=False, default=50, help="Window size from peak to left")
ap.add_argument("-ws_r", "--window_size_r", required=False, default=50, help="Window size from peak to right")
ap.add_argument("-mi", "--max_isi", required=False, default=-1, help="Maximum value for an ISI in ms (p.e. 900ms)")
ap.add_argument("-sh", "--show", required=False,default='y', help="Show plot")
ap.add_argument("-sv", "--save", required=False,default='y', help="Save events")

args = vars(ap.parse_args())

path = args['path']
scale = float(args['scale'])
dt = float(args['time_step'])

ws_l = int(args['window_size_l'])
ws_r = int(args['window_size_r'])
ws = ws_l + ws_r

tol = float(args['tol'])
print(tol)

max_isi = float(args['max_isi'])

show= True if args['show']=='y' else False 
save= True if args['save']=='y' else False 


col = int(args['column'])
ext = args['extension']

	
try:
	#WARNING!!!! skiprows 2 in case header is 2 lines. 
	df = pd.read_csv(path, delimiter = " ",skiprows=2,header=None)
except:
	print("Error: file not found",path)
	exit()

print("Getting spikes from ",path)

data = df[col]

#WARNING!!!!!! spikes not detected from RTXI bad scale!
data = data*scale #Signal should be in mV


print(data.shape)
# data = data.values


# print(data.shape)
# # print(type(data))

# denoise = True

# # beta: denoise signal
# if denoise:
# 	Wn = 0.0085
# 	N=5
# 	b, a = signal.butter(N, Wn, 'low')
# 	data = signal.filtfilt(b, a, data)

# print(data.shape)
# print(type(data))
# plt.plot(data)
# plt.plot(d_data)
# plt.show()

# print(data)
# print(data.shape)
# data = data +50

spikes_onoff,th = utils.detect_spikes(data,dt=dt,tol=tol) #Tolerance for signal in mV
spikes_single,spikes_single_v = utils.detect_spikes_single_events(data,dt,tol)

# print(type(spikes_single),type(spikes_single_v))

time = np.arange(0,data.shape[0],1)*dt


# print(spikes_onoff)
print(spikes_onoff.shape)
print(spikes_single.shape)
print(spikes_single_v.shape)


if spikes_onoff.shape[0] == 0:
	print("No spike detected")
	exit()

# isis = utils.get_ISI(spikes)

# spikes_select = np.delete(spikes,np.where(isis<=min(isis)+1))
# spikes_select = spikes

# plt.figure(figsize=(20,15))
# plt.plot(time,data)
# plt.plot(spikes_select,np.ones(spikes_select.shape)*th,'.')
# plt.show()

plt.figure(figsize=(20,15))
plt.plot(time,data)
plt.plot(spikes_onoff,np.ones(spikes_onoff.shape)*th,'.')
plt.plot(spikes_single,spikes_single_v,'.')

if ext != "":
	ext = '_' + ext

indx=path.rfind("/")
if indx < 0:
	print("Error in path, either absolute or starting with ./ path")
	exit()

events_path = path[:indx]+"/events/"

os.system("mkdir -p %s"%events_path)

if show:
	plt.show()
else:
	plt.savefig(events_path+path[indx:-4]+ext+"_spikes_detected.png")


if max_isi > 0:
	isis = np.array(utils.get_ISI(spikes_single))
	isis_onoff = np.array(utils.get_ISI(utils.to_mean(utils.to_on_off_events(spikes_onoff))))

	reduced_events = spikes_single[np.where(isis < max_isi)]
	reduced_events_indx = np.where(isis > max_isi)
	reduced_events_indx_onoff = np.where(isis_onoff > max_isi)

	plt.figure(figsize=(20,15))
	plt.plot(time,data)
	plt.plot(reduced_events, np.zeros(reduced_events.shape),'x', ms=10)
	plt.plot(spikes_single, np.zeros(spikes_single.shape),'.')

	if show:
		plt.show()
	else:
		plt.savefig(events_path+path[indx:-4]+ext+"_spikes_detected_reduced.png")

	# spikes_onoff = spikes_onoff[reduced_events_indx]
	# spikes_single = spikes_single[reduced_events_indx]

else:
	reduced_events_indx = []
	reduced_events_indx_onoff = []
	reduced_events = []

print("Number of spikes onoff detected: ",spikes_onoff.shape)
print("Number of spikes detected: ",spikes_single.shape)
print("Removing %d events"%(len(reduced_events)))

if len(reduced_events) > len(spikes_single) * 0.4:
	print(np.median(isis))
	print("Aborting: trying to remove %f of spikes"%(len(reduced_events)/len(spikes_single)))
	print("Median of ISIS: %d"%np.median(isis))
	exit()

if spikes_onoff.shape[0] > 10000:
	print("Error: number of spikes over the safety threshold. Possible failure detecting spikes. Events files won't be saved.")
	exit()

if save:

	save_path_onoff_events = events_path + path[indx:-4] + "%s_events.txt"%ext
	save_path_single_events = events_path + path[indx:-4] + "%s_single_events.txt"%ext
	save_path_waveforms = events_path + path[indx:-4] + "%s_waveform.txt"%ext
	save_path_waveforms_single = events_path + path[indx:-4] + "%s_waveform_single.txt"%ext

	#TODO: fix "selec" easy way to remove spikes --> also in charac
	print("Saving events data at \t\n",save_path_onoff_events)
	utils.save_events(spikes_onoff,save_path_onoff_events,split=True, selec=reduced_events_indx_onoff)
	print("Saving events data at \t\n",save_path_single_events)
	utils.save_events(spikes_single,save_path_single_events,split=False, selec=reduced_events_indx)
	print("Saving waveform data at \t\n",save_path_waveforms)
	waveforms = utils.save_waveforms(data,spikes_onoff,save_path_waveforms,width_ms_l=ws_l,width_ms_r=ws_r,dt=dt, selec=reduced_events_indx_onoff)
	print("Saving waveform data at \t\n",save_path_waveforms_single)
	waveforms = utils.save_waveforms(data,spikes_single,save_path_waveforms_single,width_ms_l=ws_l,width_ms_r=ws_r,dt=dt, selec=reduced_events_indx, onoff=False)

