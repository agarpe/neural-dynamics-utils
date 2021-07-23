import charact_utils as utils 
import argparse 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import scipy.signal as signal
import os


ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=True, help="Path to the file to analyze")
ap.add_argument("-e", "--extension", required=True, help="File extension")
ap.add_argument("-c", "--column", required=False,default=0, help="Column")
ap.add_argument("-sc", "--scale", required=False, default=1, help="Scale for Volts. Signal*scale")
ap.add_argument("-dt", "--time_step", required=False, default=0.1, help="Time step")
ap.add_argument("-sh", "--show", required=False,default='y', help="Show plot")
ap.add_argument("-sv", "--save", required=False,default='y', help="Save events")

args = vars(ap.parse_args())

path = args['path']
scale = float(args['scale'])
dt = float(args['time_step'])

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
print(type(data))
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
spikes_onoff,th = utils.detect_spikes(data,dt=dt,tol=0.5) #Tolerance for signal in mV
spikes_single,spikes_single_v = utils.detect_spikes_single_events(data,dt,0.5)

print(type(spikes_single),type(spikes_single_v))

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

if show:
	plt.show()
else:
	plt.savefig(path[:-4]+"_"+ext+"_spikes_detected.png")

print("Number of spikes onoff detected: ",spikes_onoff.shape)
print("Number of spikes detected: ",spikes_single.shape)
if spikes_onoff.shape[0] > 10000:
	print("Error: number of spikes over the safety threshold. Possible failure detecting spikes. Events files won't be saved.")
	exit()


if save:
	indx=path.rfind("/")

	os.system("mkdir -p %s"%path[:indx]+"/events")

	save_path_onoff_events = path[:indx]+"/events"+path[indx:-4]+"_%s_events.txt"%ext
	save_path_single_events = path[:indx]+"/events"+path[indx:-4]+"_%s_single_events.txt"%ext
	save_path_waveforms = path[:indx]+"/events"+path[indx:-4]+"_%s_waveform.txt"%ext

	print("Saving events data at \t\n",save_path_onoff_events)
	utils.save_events(spikes_onoff,save_path_onoff_events,split=True)
	print("Saving events data at \t\n",save_path_single_events)
	utils.save_events(spikes_single,save_path_single_events,split=False)
	print("Saving waveform data at \t\n",save_path_waveforms)
	waveforms = utils.save_waveforms(data,spikes_onoff,save_path_waveforms,100)

