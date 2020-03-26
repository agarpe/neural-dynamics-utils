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
# f.readline()
# f.readline()
# f.readline()
# f.readline()
# f.readline()
print(headers)
f.close()

#Create new directory for events files
os.system("mkdir -p "+path[:-4])


# for neu_name in ['N1M','N2v']:
# for neu_name in ['SO']:


#clean headers
to_clean = {"t","c"}
for e in to_clean:
	if e in headers: headers.remove(e);


for neu_name in headers:
# for neu_name in headers[1:]:
# for neu_name in ['N3t']:
	#-------------------------------------------------------
	#Read and parse data
	#-------------------------------------------------------
	print("Detecting events in neuron: ",neu_name)
	#Read column by column
	data = pd.read_csv(path, delimiter = " ", names={"t",neu_name},skiprows=1)
	neuron = data[neu_name]
	time = data['t']
	time = np.array(time) 
	
	dt = time[1]-time[0]

	neuron = np.array(neuron)
	neuron = np.gradient(neuron) #Neuron gradient to ignore drift


	#-------------------------------------------------------
	#Obtain threshold as a 1/3 of max spike value
	#-------------------------------------------------------

	mx = max(abs(neuron))
	min_dif=min(((neuron[1:])-(neuron[:-1])))
	# print(min(neuron))

	th_u = mx-mx/3
	th_l = th_u+6*abs(min_dif)
	# th_l = th_u+1

	# print(mx,th_l,th_u)

	print("threshold value: ",th_u,th_l,min_dif)


	#-------------------------------------------------------
	#Get spikes := neuron values in threshold range
	#-------------------------------------------------------

	spk = time[np.where((neuron<=th_l)&(neuron>=th_u))]
	spk_save=spk
	spk = spk[np.where(abs(spk[1:]-spk[:-1])>dt*3)]
	# spk2 = time[np.where(neuron>=th_u)]

	# spk = np.union1d(spk1,spk2)
	print(spk.shape)
	plt.figure(figsize=(15,10))

	plt.plot(spk_save,np.ones(spk_save.shape)*th_u,'.')
	plt.plot(spk,np.ones(spk.shape)*th_u,'.')

	plt.plot(time,neuron,'.')

	# plt.show()

	# print(len(spk))

	#-------------------------------------------------------
	#Compute ISI and IBI
	#-------------------------------------------------------

	diff = spk[1:] - spk[:-1] #Intervals between events
	diff_sor = np.sort(diff) #Intervals sorted 
	

	# print(len(diff))
	#ignore too close events ?
	# diff = diff[np.where(diff > dt)]

	# print(len(diff))

	#-------------------------------------------------------
	#Detect 3 types of intervals --> artefact, ISI and IBI
	#	get isi_max as the previous value before IBI.
	#-------------------------------------------------------

	# plt.plot(spk,np.ones(spk.shape)*th_u,'.')
	# # plt.plot(events,np.zeros(events.shape),'.')
	# plt.plot(time,neuron)

	# plt.show()

	# diff_sor = diff_sor[np.where(diff_sor[1:]-diff_sor[:-1] > diff_sor[:-1]*1.1)]
	# print(len(diff_sor))
	intervals = []
	for inx,(d,prev) in enumerate(zip(diff_sor[1:],diff_sor[:-1])):
		# if(intervals != [] and abs(d-prev) > prev*2): #IBI section
		# 	intervals.append(d)
		# 	print("A",inx,d,diff_sor[inx-1],diff_sor[inx+1],diff_sor[-1])
		if(d > prev*1.5): #Artefact and ISI section
			# print("Solved:",d,prev)
			# inx = np.where(diff_sor==d)
			# print("B",inx,d,diff_sor[inx-1],diff_sor[inx-10:inx],diff_sor[inx+1],diff_sor[-1],diff_sor[-2])
			# print(diff_sor[-300:])
			art = prev
			isi = d
			isi_max = diff_sor[inx-1]
			if(intervals ==[]):
				intervals.append(prev)
			intervals.append(d)

	# print("artefact,ISI and IBI: ",intervals)
	art,isi,ibi = intervals[:3]
	# art,isi = intervals[:2]

	isi = isi_max
	# isi = 222.42
	print("ISI (max):",isi,"IBI (min):",ibi)

	# Get on and off events (init and end burst) 
	# 		from spikes array and intervals ISI, IBI
	events = []
	events.append(spk[0])

	# for i,p in enumerate(spk):
	# 	if(i>1 and i<spk.shape[0]-1):
	# 		if(abs(spk[i]-spk[i-1]) <= isi and abs(spk[i+1]-spk[i]) >= ibi): #Off event: ibi after isi
	# 			events.append(p)
	# 		elif(abs(spk[i]-spk[i-1]) >= ibi and abs(spk[i+1]-spk[i]) <= isi): #On event: isi after ibi
	# 			events.append(p)

	# for i,p in enumerate(spk):
	# 	if(i>1 and i<spk.shape[0]-1):
	# 		if(abs(spk[i]-spk[i-1]) <= isi and abs(spk[i+1]-spk[i]) > isi): #Off event: ibi after isi
	# 			events.append(p)
	# 		elif(abs(spk[i]-spk[i-1]) > isi and abs(spk[i+1]-spk[i]) <= isi): #On event: isi after ibi
	# 			events.append(p)
	for i,p in enumerate(spk):
		if(i>1 and i<spk.shape[0]-1):
			if(abs(spk[i]-spk[i-1]) >= ibi and abs(spk[i+1]-spk[i]) < ibi): #Off event: ibi after isi
				events.append(p)
			elif(abs(spk[i]-spk[i-1]) < ibi and abs(spk[i+1]-spk[i]) >= ibi): #On event: isi after ibi
				events.append(p)


	events =np.array(events)


	print("Events shape result:")
	print(events.shape)


	# spk = spk[np.where(diff>art)]
	#Plot result and save events file. 
	plt.figure(figsize=(15,10))
	plt.plot(spk,np.ones(spk.shape)*th_u,'.')
	plt.plot(events,np.ones(events.shape)*mx,'.')
	plt.plot(time,neuron)
	plt.savefig(path[:-4]+"/"+neu_name+".png")
	# plt.show()

	save_events(events,path[:-4]+"/"+neu_name+"_burst.txt",split=True)
	os.system("echo 'Neuron: "+neu_name+" precission: "+str(isi)+"\n' >> "+path[:-4]+"/"+"precission.txt");

