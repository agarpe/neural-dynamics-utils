

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 16})

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

import charact_utils as utils



if len(sys.argv) > 2:
	path = sys.argv[1]
	file = sys.argv[2]
	filename = path + file
else:
	path = ''
	filename  = 'test-spikes.asc'


# folder = "/home/alicia/Documentos/data/3-impulse/22-11-2019/"
# exp_ = "exp1/"

# if(len(sys.argv)>1):
#     exp_ = sys.argv[1]

# pulses = utils.read_spike_events(path + "events.txt",dt=0.001)
# spikes = utils.read_spike_events(path + "spikes.txt",dt=0.001)


f = open(filename)
headers = f.readline().split()
print(headers)
f.close()

data = pd.read_csv(filename, delimiter = " ", names=headers,skiprows=1)


act =  np.array(data['v'])
pulses =  np.array(data['pre'])
time = np.array(data['t'])
print(time)

th=-50

events_spikes = time[np.where(act > th)]

events_spikes = events_spikes[np.where((events_spikes[1:]-events_spikes[:-1]) > 0.5)]




plt.plot(events_spikes,np.zeros(events_spikes.shape),'.')
plt.plot(time,act)
plt.show()

events_pulses = time[np.where(pulses > 0.22)]

plt.plot(events_pulses,np.zeros(events_pulses.shape),'.')
plt.plot(time,pulses)
plt.show()