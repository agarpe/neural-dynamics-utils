from charact_utils import *
import sys 
import numpy as np
import pandas as pd


if len(sys.argv) >1:
	file_name = sys.argv[1]
else:
	file_name = "prueba.asc"

print("Analysing file from ",file_name)

# path = "./data/"+file_name
path = file_name

#Read header 
f = open(path)
headers = f.readline().split()
print(headers)
f.close()

#Create new directory for events files
os.system("mkdir -p "+path[:-4])


# for neu_name in ['N1M','N2v']:
# for neu_name in ['SO']:
for neu_name in ['SO','N1M','N2v','N3t']:
	#Read and parse data

	data = pd.read_csv(path, delimiter = " ", names=headers,skiprows=1)
	neuron = data[neu_name]
	time = data['t']
	time = np.array(time) 
	
	dt = time[1]-time[0]

	neuron = np.array(neuron)
	neuron = np.gradient(neuron) #Neuron gradient to ignore drift


	#Obtain threshold as a 1/3 of max spike value

	mx = max(abs(neuron))
	th_u = mx/3
	# th_l = th_u+1

	# print(mx,th_l,th_u)

	print("threshold value: ",th_u)


	#Get spikes := neuron values in threshold range
	# spk = time[np.where(neuron<th_l)]
	spk = time[np.where(neuron>th_u)]

	# print(len(spk))

	#Compute ISI and IBI

	diff = spk[1:] - spk[:-1] #Intervals between events
	diff_sor = np.sort(diff) #Intervals sorted 
	

	# print(len(diff))
	#ignore too close events ?
	diff = diff[np.where(diff > dt)]

	# print(len(diff))


	#Detect 3 types of intervals --> artefact, ISI and IBI

	intervals = []
	for d,prev in zip(diff_sor[1:],diff_sor[:-1]):
		if(intervals != [] and abs(d-prev) > prev*2): #IBI section
			intervals.append(d)
		if(abs(d-prev) > prev*10): #Artefact and ISI section
			# print("Solved:",d,prev)
			inx = np.where(diff_sor==d)
			# print(inx,diff_sor[inx[0]-1],diff_sor[inx[0]+1],diff_sor[-1])
			art = prev
			isi = d
			if(intervals ==[]):
				intervals.append(prev)
			intervals.append(d)

	print("artefact,ISI and IBI: ",intervals)
	art,isi,ibi = intervals[:3]


	# Get on and off events (init and end burst) 
	# 		from spikes array and intervals ISI, IBI
	events = []
	# events.append(spk[0])

	for i,p in enumerate(spk):
		if(i>1 and i<spk.shape[0]-1):
			if(abs(spk[i]-spk[i-1]) <= isi and abs(spk[i+1]-spk[i]) >= ibi): #Off event: ibi after isi
				events.append(p)
			elif(abs(spk[i]-spk[i-1]) >= ibi and abs(spk[i+1]-spk[i]) <= isi): #On event: isi after ibi
				events.append(p)



	events =np.array(events)


	print("Events shape result:")
	print(events.shape)


	#Plot result and save events file. 
	plt.figure(figsize=(15,10))
	plt.plot(spk,np.ones(spk.shape)*th_u,'.')
	plt.plot(events,np.zeros(events.shape),'.')
	plt.plot(time,neuron)
	plt.savefig(path[:-4]+"/"+neu_name+".png")
	plt.show()

	save_events(events,path[:-4]+"/"+neu_name+"_burst.txt",split=True)

