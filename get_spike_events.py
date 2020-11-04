import charact_utils as utils 
import argparse 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats


ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=True, help="Path to the file to analyze")
ap.add_argument("-e", "--extension", required=True, help="File extension")
ap.add_argument("-c", "--column", required=False,default=0, help="Column")
ap.add_argument("-s", "--show", required=False,default=True, help="Show plot")

args = vars(ap.parse_args())

path = args['path']
if(args['show']=='False'):
	show = False
else:
	show = True

col = int(args['column'])
ext = args['extension']


try:
	df = pd.read_csv(path, delimiter = " ",skiprows=1,header=None)
except:
	print("Error: file not found",path)
	exit()

print("Getting spikes from ",path)

data = df[col]
spikes = utils.detect_spikes(data,tol=0.1)
time = np.arange(0,data.shape[0],1)*0.1

# print(data.shape)
# print(time.shape)
# print(spikes.shape)

isis = utils.get_ISI(spikes)

spikes_select = np.delete(spikes,np.where(isis<=min(isis)+1))

isis2 = utils.get_ISI(spikes_select)
# print(min(isis))
# print(min(isis2))


# plt.figure(figsize=(20,15))
# plt.plot(spikes,np.ones(spikes.shape[0]),'.')
# plt.plot(spikes_select,np.ones(spikes_select.shape[0]),'.','r')
# plt.show()

plt.figure(figsize=(20,15))
plt.plot(time,data)
plt.plot(spikes_select,np.ones(spikes_select.shape)*-4,'.')
if show:
	plt.show()


indx=path.rfind("/")

save_path_events = path[:indx]+"/events"+path[indx:-4]+"_%s_events.txt"%ext
save_path_waveforms = path[:indx]+"/events"+path[indx:-4]+"_%s_waveform.txt"%ext

print("Saving events data at \t\n",save_path_events)
utils.save_events(spikes,save_path_events,split=True)
print("Saving waveform data at \t\n",save_path_waveforms)
waveforms = utils.save_waveforms(data,spikes,save_path_waveforms,100)

