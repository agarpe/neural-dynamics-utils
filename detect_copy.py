from charact_utils import *
import sys 
import numpy as np
import pandas as pd


if len(sys.argv) >1:
	file_name = sys.argv[1]
else:
	file_name = "test.asc"

print("Analysing file from ",file_name)

# path = "./data/"+file_name
path = file_name


f = open(path)
headers = f.readline().split()
print(headers)
f.close()

# data = np.loadtxt(path,skiprows=1)

params = {}
params['N1M'] = [-30,-31,40,90000]
params['N2v'] = [-50,-51,0,30000]
params['N3t'] = [-30,-31,20,160000]


neu_name = 'N1M'




data = pd.read_csv(path, delimiter = " ", names=headers,skiprows=1)

neuron = data[neu_name]
time = data['t']
time = np.array(time) 
time/=0.01
neuron = np.array(neuron)

th_l = params[neu_name][0]
th_u =  params[neu_name][1]

# events = time[np.where(neuron<th_l)]
# events = time[np.where(neuron>th_u)]
events = []

prev = 0
for p,n in zip(time,neuron):
	if (n < th_l and n> th_u and abs(p-prev) > params[neu_name][3] ):
		events.append(p)
		prev = p

events =np.array(events)





# plt.plot(events,np.zeros(events.shape),'.')
# plt.plot(time,neuron)
# plt.show()

spikes=[]
for i in range(0,events.shape[0]-2,2):
	if(abs(events[i]-events[i+1])< 20000):
		spikes.append((events[i]+events[i+1])//2)

spikes = np.array(spikes)
# spikes = np.array([(a+b)//2 for i,j in zip(range(,events[1:]))
print(spikes.shape)
# spikes = to_mean(events)


plt.figure(figsize=(15,10))
plt.plot(events,np.zeros(events.shape),'.')
plt.plot(spikes,np.zeros(spikes.shape),'.',color='red')
plt.plot(time,neuron)
plt.show()


prev=0
burst=[spikes[0]]
for s in spikes:
	if s - prev > params[neu_name][3]:
		burst.append(s)
		prev = s

burst=np.array(burst)


plt.plot(spikes,np.zeros(spikes.shape),'.')
plt.plot(burst,np.ones(burst.shape),'.',color='red')
plt.plot(time,neuron)
plt.show()