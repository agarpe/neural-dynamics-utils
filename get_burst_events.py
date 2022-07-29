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
ap.add_argument("-sc", "--scale", required=False, default=1, help="Scale for Volts. Signal*scale")
ap.add_argument("-misi", "--misi", required=False, default=0, help="Max isi value estimated in ms")
ap.add_argument("-dt", "--time_step", required=False, default=0.1, help="Time step")
ap.add_argument("-sh", "--show", required=False,default='y', help="Show plot")
ap.add_argument("-sv", "--save", required=False,default='y', help="Save events")

args = vars(ap.parse_args())

path = args['path']
scale = float(args['scale'])
tol = 0.5*scale
# tol = 0.5/scale
dt = float(args['time_step'])
max_isi = float(args['misi'])

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

# max_isi = 100

bursts,th = utils.detect_burst_from_signal(data,max_isi,dt,tol=tol)


time = np.arange(0,data.shape[0],1)*dt
print(bursts.shape)
# print(bursts[:,0])

plt.figure(figsize=(20,10))
plt.plot(bursts[:,0],np.ones(bursts.shape)*th,'.')
plt.plot(bursts[:,1],np.ones(bursts.shape)*th,'.')
plt.plot(time,data)


if show:
	plt.show()
else:
	plt.savefig(path[:-4]+"_"+ext+"_burst_detected.png")


# print("Number of spikes detected: ",spikes_select.shape[0])
# if spikes_select.shape[0] > 10000:
# 	print("Error: number of spikes over the safety threshold. Possible failure detecting spikes. Events files won't be saved.")
# 	exit()


if save:
	indx=path.rfind("/")

	save_path_events = path[:indx]+"/events"+path[indx:-4]+"_%s_burst_events.txt"%ext
	# save_path_waveforms = path[:indx]+"/events"+path[indx:-4]+"_%s_waveform.txt"%ext

	print("Saving events data at \t\n",save_path_events)
	utils.save_events(bursts,save_path_events,split=False)
	# print("Saving waveform data at \t\n",save_path_waveforms)
	# waveforms = utils.save_waveforms(data,spikes_select,save_path_waveforms,100)

