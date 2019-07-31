import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os


#############################################################################
##############	PLOT 
##############################################################################

##Plots isi and zoom in by xlim and y lim 
def plot_return_map(ISI,title,xlim=(10,50),ylim=(10,50),outliers=1):
    plt.title(title+" ("+str(len(ISI))+")")
    plt.plot(ISI[:-outliers],ISI[outliers:],'.',markersize=1)
    plt.plot(ISI,ISI,linewidth=0.3)
    plt.xlabel('ISI_i [s]')
    plt.ylabel('ISI_i+1 [s]')
    plt.show()
    
    plt.title(title+" ("+str(len(ISI))+")")
    plt.xlim(xlim[0],xlim[1])
    plt.ylim(ylim[0],ylim[1])
    plt.plot(ISI[:-1],ISI[1:],'.',markersize=1)
    plt.plot(ISI,ISI,linewidth=0.3)
    plt.xlabel('ISI_i [s]')
    plt.ylabel('ISI_i+1 [s]')
    plt.show()


def plot_hists(charac,neuron):

	plt.subplot(1,3,1)
	plt.title("Burst duration " + neuron)
	plt.hist(charac[DUR],rwidth=0.4)
	plt.xlabel("Time (s)")


	plt.subplot(1,3,2)
	plt.title("Burst Interval " + neuron)
	plt.hist(charac[IBI],rwidth=0.4)
	plt.xlabel("Time (s)")

	plt.subplot(1,3,3)
	plt.title("Burst Period " + neuron)
	plt.hist(charac[PER],rwidth=0.4)
	plt.xlabel("Time (s)")
	plt.show()


def plot_corr(elem1,elem2,title1,title2):

	plt.plot(elem1,elem2,'.')
	plt.xlabel(title1)
	plt.ylabel(title2)
	plt.show()




#############################################################################
##############	READ, WRITE PLOT 
##############################################################################


def save_events(events,file_name):

	f1 = open(file_name,'w')
	np.savetxt(f1,events,delimiter='\t')
	f1.close()
	#changes . by , as separator (for dataview)
	os.system("sed -i 's/\./,/g' "+file_name)


def read_spike_events(file_name):

	data_n = np.loadtxt(file_name)

	print(data_n.shape)

	#Change to secs

	data_n /= 1000

	#get half

	data_n = data_n[:(data_n.shape[0]//1)]

	#Gets spikes as mean from up off events. 
	mean_evt_n = to_mean(data_n)
	print(mean_evt_n.shape)
	print(mean_evt_n[:4],mean_evt_n[-1])

	return mean_evt_n

#############################################################################
##############	SPIKES 
##############################################################################


##spike condition --> mean point init-end event
def to_mean(data):
    return np.array([np.mean([a,b]) for a,b in zip(data[:,0],data[:,1])])

##Gets isi as difference between each 2 events
def get_ISI(events):
	isi = list(map(lambda x,y: y-x, events[:-1],events[1:]))
	return isi

#Computes Spike Density convoluting frequencies with gaussian 
def sdf(spikes,window_size=3,sigma=2):
	filt = signal.gaussian(window_size, sigma)
	# plt.plot(filt)
	# plt.show()
	return signal.convolve(spikes,filt,mode='full') #full: discrete linear convolution
											 #same: same size as in1, centered with respect to the full

#Generates an array [0,1] where 0 := no event; 1 := event
def get_spikes(events):
	dt =0.001
	N=int((events[-1]+0.1)/dt)
	print(N)
	act = np.full(N,0)

	for e in events:
		act[int(e/dt)] =1
	# act[np.where(events!=act)] =0

	return act


##Detects derivate difference
#### if 500 points decreasing, 500 increasing --> positive p
def diff(signal,n=500):
	count_desc= 0
	count_asc = 0
	prev = 0
	for s,prev in zip(signal[:,-1],signal[1,:]):
		if prev < s:
			count_asc +=1


# n := number of points determining increase decrease
#output: events array [t,s]
def events_from_thres(signal,n=4):
	events = []
	state = -1
	count = 0
	aux = []
	for i,s in enumerate(signal[1:]):
		index = i + 1

		if(signal[index] >= signal[index-1]): #increasing
			if state == -1: #decreasing 
				state = 1
				if len(events)> 0:
					if abs(s - events[-1][1]) >=n: #number of points between events > n
						events.append((index,s))
				else:
					events.append((index,s))
				count = 0	

			else: 
				count +=1 #time increasing
		else:
			if state == 1: #decreasing 
				state = -1
				if len(events)>0:
					if abs(s - events[-1][1]) >=n: #number of points between events > n
						events.append((index,s))
				else:
					events.append((index,s))
				count = 0	
			else: 
				count +=1 #time decreasing
		aux.append(count)

	# plt.hist(aux)
	# plt.show()
	# print(max(aux),min(aux))

	return np.array(events)


def get_phases_from_events(events,n_phases=3):
	phases = []
	for i in range(n_phases):
		phases.append([])

	print(phases)

	for i,e in enumerate(events):
		# for p in range(n_phases):
		try:
			# print(phases[p])
			# print(events[i+p])
			
			phases[i%n_phases].append((events[i],events[i+1]))
			# print(phases[p])
		except:
			pass
	print(np.array(phases[0]).shape)


	return phases



def get_phases(data,init,end,th1=6,th2=7.5):
	phase = 3
	phase1 =[]
	phase2 =[]
	phase3 = []
	for i,p in enumerate(data):
		t = init+i
		if abs(th1-p) < 0.00001:
			if len(phase1) == 0:
				phase1.append((t,p))
			elif (t-phase1[-1][0]) > 500:
				phase1.append((t,p))
					

			# phase = 1
			# print(p,i,th1-p)
			# print("a")

		elif abs(th2-p) < 0.00001:
			if len(phase3) == 0:
				phase3.append((t,p))
			elif (t-phase3[-1][0]) > 500:
				phase3.append((t,p))
			# phase1.append((i,p))
			# phase = 2
			# print(p,i,p-th2)
			# print("b")


	# phase1 = data[np.where(data-2.0 > 0.1)]
	return np.array(phase1),np.array(phase3)

#############################################################################
##############	BURSTS 
##############################################################################


##off1 - on1 
def get_burst_duration(data):
    return np.array([b-a for a,b in zip(data[:,0],data[:,1])])

##on2 - off1
def get_burst_interval(data):
	return np.array([a-b for a,b in zip(data[1:,0],data[:,1])])


##on2 - on1
def get_burst_period(data):
	return np.array([a-b for a,b in zip(data[1:,0],data[:,0])])


DUR = 0
IBI = 1
PER = 2


def analyse(data):
	dur = get_burst_duration(data)

	ibi = get_burst_interval(data)

	period = get_burst_period(data)

	return dur,ibi,period



###?¿? = analyse

def analyse_hists(data,neuron,plot=True):
	charac = analyse(data)

	if plot:
		plot_hists(charac,neuron)

	return charac

