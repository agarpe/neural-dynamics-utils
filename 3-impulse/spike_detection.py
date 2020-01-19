

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
else:
	path = ''
	file  = 'test-spikes.asc'

filename = path + file

# folder = "/home/alicia/Documentos/data/3-impulse/22-11-2019/"
# exp_ = "exp1/"

# if(len(sys.argv)>1):
#     exp_ = sys.argv[1]

# pulses = utils.read_spike_events(path + "events.txt",dt=0.001)
# spikes = utils.read_spike_events(path + "spikes.txt",dt=0.001)


f = open(filename)
headers = f.readline().split()
# print(headers)
f.close()

data = pd.read_csv(filename, delimiter = " ", names=headers,skiprows=1)

#init = 2000
#end = 2000
init = 0
end = 0

act =  np.array(data['v'])[init:-end]
pulses =  np.array(data['pre'])[init:-end]

if('t' not in data.keys()):
	dt = 0.0001
	max_ = act.shape[0] * dt
	time = np.array(np.arange(0,max_,dt))
else:
	time = np.array(data['t'])[init:-end]
# print(time)

th=-40

events_spikes = time[np.where(act > th)]

# print(events_spikes.shape)
events_spikes = events_spikes[np.where((events_spikes[1:]-events_spikes[:-1]) > 0.2)]
# print(events_spikes.shape)
# print(events_spikes[:10])


plt.plot(events_spikes[2:],np.zeros(events_spikes[2:].shape),'.')
plt.plot(time,act)
plt.show()
plt.close()
events_pulses = time[np.where(pulses > 0.3)]

plt.plot(events_pulses,np.zeros(events_pulses.shape),'.')
plt.plot(time,pulses)
plt.show()
plt.close()

dirname = file[:-4]


output = os.system("mkdir -p "+ filename[:-4])

np.savetxt(filename[:-4]+"/spikes.txt",events_spikes[2:])
np.savetxt(filename[:-4]+"/events.txt",events_pulses[1:])
# np.savetxt(filename[:-4]+"/prueba.txt",(events_spikes,events_pulses))

