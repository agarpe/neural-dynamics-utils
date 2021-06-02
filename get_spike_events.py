import charact_utils as utils 
import argparse 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import scipy.signal as signal


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

print("Getting spikes from ",path)

data = df[col]

#WARNING!!!!!! spikes not detected from RTXI bad scale!
data = data*10


denoise = True

#beta: denoise signal
if denoise:
	Wn = 0.0085
	N=5
	b, a = signal.butter(N, Wn, 'low')
	data = signal.filtfilt(b, a, data)

# plt.plot(data)
# plt.plot(d_data)
# plt.show()

# print(data)
# print(data.shape)
spikes,th = utils.detect_spikes(data,tol=1)
time = np.arange(0,data.shape[0],1)*0.1


isis = utils.get_ISI(spikes)

spikes_select = np.delete(spikes,np.where(isis<=min(isis)+1))

isis2 = utils.get_ISI(spikes_select)

plt.figure(figsize=(20,15))
plt.plot(time,data)
plt.plot(spikes_select,np.ones(spikes_select.shape)*th,'.')

if show:
	plt.show()
else:
	plt.savefig(path[:-4]+"_"+ext+"_spikes_detected.png")


print("Number of spikes detected: ",spikes_select.shape[0])
if spikes_select.shape[0] > 10000:
	print("Error: number of spikes over the safety threshold. Possible failure detecting spikes. Events files won't be saved.")
	exit()


if save:
	indx=path.rfind("/")

	save_path_events = path[:indx]+"/events"+path[indx:-4]+"_%s_events.txt"%ext
	save_path_waveforms = path[:indx]+"/events"+path[indx:-4]+"_%s_waveform.txt"%ext

	print("Saving events data at \t\n",save_path_events)
	utils.save_events(spikes_select,save_path_events,split=True)
	print("Saving waveform data at \t\n",save_path_waveforms)
	waveforms = utils.save_waveforms(data,spikes_select,save_path_waveforms,100)

