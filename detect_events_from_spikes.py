from charact_utils import *
import sys 
import numpy as np
import pandas as pd

def parse_spikes(spk_time,spk_data):
	print(spk_time.shape,spk_data.shape)
	return spk_time[np.where(spk_data)]

def parse_path(path):
	if "spikes" in path: 
		path_spk = path
	else:
		try:
			idx = path.index("Euler")
		except:
			idx = path.index("Runge")

		path_spk = path[:idx] + "spikes_" + path[idx:]

		print(path_spk)
	return path_spk

def get_file_info(path):
	f = open(path_spk)
	no_spike_value = float(f.readline())
	headers = f.readline().split()
	print(headers)
	f.close()

	return headers,no_spike_value

def read_spikes(path,headers,indexes,no_spike_value):
	print(headers)
	print(indexes)
	spikes = pd.read_csv(path_spk, delimiter = " ", names=headers,skiprows=2,low_memory=False,na_values=no_spike_value,usecols=indexes)

	return spikes.dropna()


if len(sys.argv) >1:
	path = sys.argv[1]
else:
	print("detect_events_from_spikes.py file_path")
	exit()

print("Analysing file from ",path)


path_spk = parse_path(path)

headers,no_spike_value = get_file_info(path_spk)


#Create new directory for events files
os.system("mkdir -p "+path[:-4])

neu_headers = headers

#clean headers
to_clean = {"t","c"}
for e in to_clean:
	if e in headers: neu_headers.remove(e);


for index,neu_name in enumerate(neu_headers):

	#-------------------------------------------------------
	#Read and parse data
	#-------------------------------------------------------
	print("Detecting events in neuron: ",neu_name)
	#Read column by column
	# data = pd.read_csv(path, delimiter = " ", names={"t",neu_name},skiprows=1)
	data = read_spikes(path_spk,['t',neu_name],[0,index+1],",")
	neuron = data[neu_name]
	time = data['t']
	time = np.array(time) 

	print(neuron[:10])
	#-------------------------------------------------------
	#Get spikes in form of time events
	#-------------------------------------------------------

	spk = parse_spikes(time, neuron);

	print(spk.shape)

	#-------------------------------------------------------
	#Compute ISI and IBI
	#-------------------------------------------------------

	diff = spk[1:] - spk[:-1] #Intervals between events
	diff_sor = np.sort(diff) #Intervals sorted 
	
	intervals = []
	for inx,(d,prev) in enumerate(zip(diff_sor[1:],diff_sor[:-1])):
		if(d > prev*2): #ISI to IBI
			isi = d
			isi_max = diff_sor[inx-1]
			if(intervals ==[]):
				intervals.append(prev)
			intervals.append(d)

	isi,ibi = intervals[:2]

	isi = isi_max
	# isi = 222.42
	print("ISI (max):",isi,"IBI (min):",ibi)

	# Get on and off events (init and end burst) 
	# 		from spikes array and intervals ISI, IBI
	events = []
	events.append(spk[0])

	for i,p in enumerate(spk):
		if(i>1 and i<spk.shape[0]-1):
			if(abs(spk[i]-spk[i-1]) >= ibi and abs(spk[i+1]-spk[i]) < ibi): #Off event: ibi after isi
				events.append(p)
			elif(abs(spk[i]-spk[i-1]) < ibi and abs(spk[i+1]-spk[i]) >= ibi): #On event: isi after ibi
				events.append(p)


	events =np.array(events)


	print("Events shape result:")
	print(events.shape)


	#Plot result and save events file. 
	plt.figure(figsize=(15,10))
	plt.plot(spk,np.ones(spk.shape),'.')
	plt.plot(events,np.ones(events.shape),'.')
	plt.savefig(path[:-4]+"/"+neu_name+".png")
	plt.show()

	save_events(events,path[:-4]+"/"+neu_name+"_burst.txt",split=True)
	# os.system("echo 'Neuron: "+neu_name+" precission: "+str(isi)+"\n' >> "+path[:-4]+"/"+"precission.txt");

