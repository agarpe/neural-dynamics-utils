from charact_utils import *
import sys 
import numpy as np
import pandas as pd

from matplotlib import cm
from colorspacious import cspace_converter
from collections import OrderedDict

cmaps = OrderedDict()

if len(sys.argv) >1:
	path = sys.argv[1]
else:
	path = "prueba.asc"

f = open(path)
headers = f.readline().split()
print(headers)
f.close()

data = pd.read_csv(path, delimiter = " ", names=headers,skiprows=1)


colors = ['maroon', 'teal', 'brown', 'blue', 'green']

figure_handle = plt.figure(figsize=(30,20))
for i,neu_name in enumerate(headers[1:]):
	neuron = data[neu_name]
	time = data['t']
	time = np.array(time) 
	# time*=0.01
	neuron = np.array(neuron)

	if(i==0):
		ax1 = plt.subplot(len(headers)-1,1,i+1)
	else :
		plt.subplot(len(headers)-1,1,i+1,sharex=ax1)

	if(neu_name in ["SO","N1M","N2v","N3t"]):
		events = read_model_burst_path(path[:-4]+"/"+neu_name,scale=1)

		print(events.shape)


		plt.plot(events,np.zeros(events.shape),'.')
	plt.plot(time,neuron,color=colors[i%len(colors)])

plt.savefig(path[:-4]+".png")
# plt.show()