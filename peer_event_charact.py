import charact_utils as utils 
import argparse 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import scipy.signal as signal


# def plot_hist()

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=True, help="Path to the file to analyze")
# ap.add_argument("-p2", "--path2", required=True, help="Path to the file to analyze")
ap.add_argument("-r","--range",required=False,default="-15,0", help="Range of data")
ap.add_argument("-onoff", "--onoff", required=False,default='y', help="Events as on-off")
ap.add_argument("-sh", "--show", required=False,default='y', help="Show plot")
ap.add_argument("-sv", "--save", required=False,default='y', help="Save events")
args = vars(ap.parse_args())


path = args['path']

indx=path.rfind("/")
title = path[indx+1:]

path1 = path+"_RPD2_events.txt"
path2 = path+"_VD1_events.txt"

rang = [float(r) for r in args['range'].split()]
print(rang)

show= True if args['show']=='y' else False 
save= True if args['save']=='y' else False 
onoff=True if args['onoff']=='y' else False 


spikes1 = utils.read_spike_events(path1,dataview=onoff)
spikes2 = utils.read_spike_events(path2,dataview=onoff)



if spikes1.shape != spikes2.shape:
	print("Error: the number of spikes is different")
	plt.plot(spikes1,np.ones(spikes1.shape),'.')
	plt.plot(spikes2,np.ones(spikes2.shape),'.')
	print(spikes1.shape,spikes2.shape)
	plt.show()
	exit()

diff = spikes1-spikes2

plt.figure(figsize=(10,10))
plt.hist(diff,width=0.1)
plt.title(title)
plt.xlabel("Event RPD2-VD1 (ms)")

save_path = path+"_peer_event_hist"
print(save_path)
plt.savefig(save_path+".eps",format='eps')
plt.savefig(save_path+".png",format='png')
plt.show()


plt.figure(figsize=(10,10))
plt.hist(diff,range=rang,width=0.7)
plt.title(title)
plt.xlabel("Event RPD2-VD1 (ms)")

# indx=path1.rfind("RPD2")
save_path = path+"_peer_event_hist_range"
print(save_path)
plt.savefig(save_path+".eps",format='eps')
plt.savefig(save_path+".png",format='png')
plt.show()