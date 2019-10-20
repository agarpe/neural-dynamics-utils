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


f = open(path)
headers = f.readline().split()
print(headers)
f.close()

# data = np.loadtxt(path,skiprows=1)

params = {}
params['SO'] = [-30,-31]
params['N1M'] = [-30,-31]
params['N2v'] = [-45,-45]
params['N3t'] = [-30,-31]


# os.system("mkdir "+path[:-4])


for neu_name in ['SO','N1M','N2v','N3t']:
	data = pd.read_csv(path, delimiter = " ", names=headers,skiprows=1)

	neuron = data[neu_name]
	time = data['t']
	time = np.array(time) 
	time*=0.01
	neuron = np.array(neuron)


	th_l = params[neu_name][0]
	th_u =  params[neu_name][1]

	spk = time[np.where(neuron<th_l)]
	spk = time[np.where(neuron>th_u)]


	events = []

	diff = spk[1:] - spk[:-1] #Intervals between events
	diff_sor = np.sort(diff) #Intervals sorted 

	diff = diff[np.where(diff > 0.01)]


	###TODO: mejorar, max(diff)/3 valor aleatorio
	m = min(diff) + max(diff)/3
	print(min(diff),(max(diff)-min(diff))/2,m)


	maxs = diff[np.where(diff > m)]
	mins = diff[np.where(diff < m)]

	print(maxs)
	print(mins)

	ibi = min(maxs)
	mx2 = max(maxs)
	isi = max(mins)

	print(isi,ibi)


	prev_in = isi
	prev = 0
	events.append(spk[0])


	for i,p in enumerate(spk):
		if(i>1 and i<spk.shape[0]-1):
			if(abs(spk[i]-spk[i-1]) <= isi and abs(spk[i+1]-spk[i]) > isi): #Off event
				events.append(p)
			elif(abs(spk[i]-spk[i-1]) > isi and abs(spk[i+1]-spk[i]) <= isi): #On event 
				events.append(p)

	events =np.array(events)


	plt.figure(figsize=(15,10))
	plt.plot(events,np.zeros(events.shape),'.')
	plt.plot(time,neuron)
	plt.savefig(path[:-4]+"/"+neu_name+".png")
	plt.show()

	save_events(events,path[:-4]+"/"+neu_name+"_burst.txt",split=True)

