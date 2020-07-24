from superpos_functions import *


if len(sys.argv) >1:
	path = sys.argv[1]
	# path_spk = sys.argv[2]
else:
	exit()


def get_events(f_data,f_events,ms):
	#read data
	data = pd.read_csv(f_data, delimiter = " ",skiprows=1,header=None)
	data = data.values

	#read events
	events = pd.read_csv(f_events, delimiter = " ",skiprows=2,header=None)
	events = events.values
	points = int(ms /0.001)

	waveforms = np.empty((events.shape[0],points*2),float)
	print(waveforms.shape)
	print(events.shape)

	# print(points)

	time = data[:,0]

	# waveforms = np.array([[]])
	#for each event get data+- ms
	# print(events)
	count =0
	for i,event in enumerate(events[:,0]):
		# print(data[:,0])
		# print(event)
		indx = np.where(time == event)[0][0] #finds spike time reference

		# print(data[indx-points:indx+points,1].shape)
		# print(indx)
		# waveforms = np.append(waveforms,data[indx-points:indx+points,1],axis=1) #read waveform form that spike. 
		try:
			waveforms[i] =data[indx-points:indx+points,1]
		except:
			count +=1
		# waveforms.append(data[indx-points:indx+points,1]) #read waveform form that spike. 

	print(count, "events ignored")
	# print(waveforms)
	return waveforms


files = os.listdir(path)

axs = []
labels = []

for f in files:
	print(f)
	f = path+f
	if(f.find("spikes")==-1):

		print(f)
		ref = f.find("Euler")
		f_events = f[:ref]+"spikes_"+f[ref:]
		print(f_events)


		index =f.find(".asc")
		ini = f.rfind("_")

		label = "tau="+f[ini+1:index]
		print(label)
		labels.append(label)


		trial = get_events(f,f_events,10)

		ax,ax1,ax2=plot_events(trial,'#%06X' % randint(0, 0xFFFFFF),tit='Model',ms=10,dt=0.001)
		axs.append(ax)


plt.legend(axs,labels)
plt.show()






