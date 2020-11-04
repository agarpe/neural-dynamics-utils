import numpy as np
import matplotlib.pyplot as plt
#27 parece la mejor para plots. 
plt.rcParams.update({'font.size': 25})
from matplotlib import colors as mcolors
import statistics 
from scipy import signal
import os
from sklearn.linear_model import LinearRegression
from scipy.stats import linregress
import sys


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

#########################
#			#			#
#	  0		#	  1		#
#			#			#
#########################
#			#			#
#	  2		#	  3		#
#			#			#
#########################

def get_pos(pos,ran_x,ran_y):

	if(not ran_x or not ran_y): return
	x_min,x_max = ran_x[:]
	y_min,y_max = ran_y[:]

	width = x_max-x_min
	length = y_max-y_min

	if pos==0:
		x = x_min + width/4
		y = y_max - length/4
	elif pos ==1:
		x = x_max - width/4
		y = y_max - length/4
	elif pos ==2:
		x = x_min + width/4
		y = y_min + length/4
	elif pos ==3:
		x = x_max - width*0.6
		y = y_min + length/4
	else:
		x=0
		y=0

	return x,y





#text_pos inverts 
def plot_corr(x,y,title1,title2,ran_x,ran_y,show=True,color='b',text_pos_y=0,text_pos_x=0,text_pos=0):

	r_sq,Y_pred,slope = do_regression(x,y,title2,False)

	if(ran_y != False):
		max_ = ran_y[1]
		plt.ylim(ran_y)
		# plt.text(max_-(0.25*max_)*text_pos_x,max_-(0.25*max_)*text_pos_y,"R² = "+"{:6f}".format(r_sq))
		x_text,y_text = get_pos(text_pos,ran_x,ran_y)
		plt.text(x_text-text_pos_x,y_text-text_pos_y,"R² = "+"{:.4f}".format(r_sq))
	if(ran_x!= False):
		plt.xlim(ran_x)
	plt.title(title2[:-4])
	#maroon
	plt.plot(x, Y_pred, color='grey')
	plt.plot(x,y,'.',color=color)
	# plt.text(1,max_-0.25*max_,"R² = "+str(r_sq)[:8])
	# plt.text(1,max_-0.5*max_,"slope = "+str(slope)[:8])
	
	plt.xlabel(title1)
	plt.ylabel(title2)
	if show:
		plt.show()

def do_regression(x,y,title,show=True):

	slope, intercept, r_value, p_value, std_err = linregress(x,y)
	model = LinearRegression(fit_intercept=True)
	x = x.reshape((-1, 1))

	model.fit(x,y)

	r_sq = model.score(x, y)

	
	# slope, intercept, r_value, p_value, std_err = linregress([x[0],x[1]],[y[0],y[1]])
	# print(slope,intercept)
	print(title)
	print('\tCV:',np.std(y)/np.mean(y))
	print('\tmedian:',statistics.median(y))
	print('\tcoefficient of determination:', r_sq)
	# print('\tslope:', slope)

	Y_pred = model.predict(x)  # make predictions

	return r_sq,Y_pred,slope

def get_global_color_list():
	colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

	by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name)
	                for name, color in colors.items())
	

	sorted_names = [name for hsv, name in by_hsv]
			#dict and keys
	return colors,sorted_names



from sklearn.preprocessing import normalize


def plot_intervals_stats(stats,box_ran,norm=False,pos=False):
	keys = sorted(stats.keys())
	intervals = []
	labels = []

	colors_map={"Period":'coral',"BD":'royalblue',"Interval":'seagreen',"Delay":'brown'}


	colors =[]
	medians=[]
	means=[]
	cvs=[]
	for key in keys:
		elem = stats[key]
		for e in reversed(sorted(elem.keys())):
			if(key[1:] != "N1M" and e == "Period"):
				pass
			elif(e != "IBI"):
				labels.append(key[1:]+"-"+e)
				if(norm):
					if(pos):
						# interval = np.absolute(stats[key][e])
						intervals.append(np.absolute(stats[key][e])/np.linalg.norm(stats[key][e]))
						# plt.plot()
						# plt.show()
					else:
						intervals.append(stats[key][e]/np.linalg.norm(stats[key][e]))
				else:
					intervals.append(stats[key][e])
					# intervals.append(np.std(stats[key][e])/np.mean(stats[key][e])*100)					
					cvs.append((np.std(stats[key][e])/np.mean(stats[key][e]))*100)
					medians.append(statistics.median(stats[key][e])*100)
					means.append(np.mean(stats[key][e])*100)
				colors.append(colors_map[e])


	plt.figure(figsize=(33,20))
	bp = plt.boxplot(intervals, showfliers=False,labels=labels,patch_artist=True)
	# plt.scatter(range(1,len(cvs)+1),cvs)
	# plt.scatter(range(1,len(cvs)+1),medians)
	# plt.scatter(range(1,len(cvs)+1),means)
	# for patch,color in zip(bp['boxes'],sorted_names[10:]):
	used =[]
	legends=[]
	for patch,color in zip(bp['boxes'],colors):
		patch.set_facecolor(color)
		if(color not in used):
			used.append(color)
			legends.append([patch,list(colors_map.keys())[list(colors_map.values()).index(color)]])

	for patch in bp['medians']:
		plt.setp(patch, color='black',linewidth=1.5)

	legends = np.array(legends)

	plt.tick_params(axis='both', labelsize=40)
	plt.xticks(rotation=45, ha='right')
	plt.ylim(box_ran)
	plt.ylabel("Time intervals (s)",fontsize=50)

	plt.legend(legends[:,0],legends[:,1],fontsize='x-large',loc='upper center', bbox_to_anchor=(0.75,1))

	# plt.legend(legends[:,0],legends[:,1],fontsize='x-large')

	# for i,item in enumerate(reversed(sorted(colors_map.keys()))):
	# 	plt.text(15, 35-1*i, item,
 #        	 backgroundcolor=colors_map[item],
 #         	color='black', weight='roman', size='x-small')
	plt.tight_layout()



def plot_bar(stats):	
	keys = sorted(stats.keys())
	intervals = []
	labels = []
	x = []
	colors=[]
	index = 1

	colors_map={"Period":'coral',"BD":'royalblue',"Interval":'seagreen',"Delay":'brown'}
	for key in keys:
		elem = stats[key]
		for e in reversed(sorted(elem.keys())):
			if(key[1:] != "N1M" and e == "Period"):
				pass
			elif(e != "IBI"):
				labels.append(key[1:]+"-"+e)
				print(key[1:]+"-"+e,np.std(stats[key][e]))
				intervals.append(np.std(stats[key][e]))
				colors.append(colors_map[e])
				index +=1

	plt.figure(figsize=(20,10))
	plt.title("Standard deviation")
	plt.bar(range(1,index),intervals,width=0.2,color=colors)
	print(labels)
	plt.xticks(range(1,index),(labels),rotation=45)
	plt.tight_layout()
	# plt.show()






#############################################################################
##############	READ, WRITE PLOT 
##############################################################################

def to_on_off_events(events):
	###TODO: iterate until off-on > ISI
	#if there is one spike "missing" at the end--> ignore it. 
	if(events.shape[0]%2!=0):
		events = events[:-1]

	events = np.array([[events[i],events[i+1]] for i in range(0,events.shape[0],2)])
	return events

#Saves events in same file than original data. 
def save_events(events,file_name,split=False,dataview=False):
	if(split):
		events = to_on_off_events(events)

	print(events.shape)
		# events = np.array(result)
	f1 = open(file_name,'w')
	np.savetxt(f1,events,delimiter='\t')
	f1.close()
	if(dataview):
		#changes . by , as separator (for dataview)
		os.system("sed -i 's/\./,/g' "+file_name)

def save_waveforms(data,events,path,width_ms,dt=0.1):
	events = to_on_off_events(events)
	mean_evt_n = to_mean(events)
	
	points = int(width_ms /dt)

	waveforms = np.empty((events.shape[0],(points*2)),float)

	time = np.arange(0,data.shape[0],1.0)
	time *=dt

	count =0
	for i,event in enumerate(events[:,0]):
		indx = np.where(time == event)[0][0] #finds spike time reference

		try:
			waveforms[i] =data[indx-points:indx+points]
		except:
			count +=1

	print(waveforms.shape)

	f1 = open(path,'w')
	np.savetxt(f1,waveforms,delimiter='\t')
	f1.close()

	return waveforms



#Read spike events from file as on/off events and returns single value from each event as mean(on/off). 

def read_spike_events(file_name,dataview=True,dt=0.1):
	if dataview:
		#changes , by . as separator (for dataview)
		os.system("sed -i 's/\,/./g' "+file_name)

	data_n = np.loadtxt(file_name)

	# print(data_n.shape)

	#Change to secs

	# data_n *= dt

	#get half

	# data_n = data_n[:(data_n.shape[0]//1)]

	if(dataview):
	#Gets spikes as mean from up off events. 
		mean_evt_n = to_mean(data_n)
	else:
		mean_evt_n = data_n
	
	# print(mean_evt_n.shape)
	# print(mean_evt_n[:4],mean_evt_n[-1])

	return mean_evt_n




#Read spike events from file as on/off events and returns single value from each event as mean(on/off). 

def read_bursts_events(file_name,dataview=True,scale= 1000):
	if dataview:
		#changes , by . as separator (for dataview)
		os.system("sed -i 's/\,/./g' "+file_name)

	data_n = np.loadtxt(file_name)

	print(data_n.shape)

	# #Change to secs

	# data_n /= scale


	return data_n


#Read spike events from file as on/off events and returns single value from each event as mean(on/off). 

def read_model_burst(neuron,dataview=True,scale= 1000):

	file_name = '../model/'+neuron+'_burst.txt'
	print(file_name)

	return read_bursts_events(file_name,dataview,scale)


#Read spike events from file as on/off events and returns single value from each event as mean(on/off). 

def read_model_burst_path(path,dataview=True,scale= 1000):

	file_name = path+'_burst.txt'
	print(file_name)
	return read_bursts_events(file_name,dataview,scale)



def trunc(values, decs=0):
    return np.trunc(values*10**decs)/(10**decs)

#############################################################################
##############	SPIKES 
##############################################################################

def detect_spikes(data,dt=0.1,tol=0.2):

	#define threshold
	mx_value = np.max(data) #maximum V value (spike)
	mn_value = np.min(data) #minimum V value (spike)

	th = (mx_value-mn_value)/2 #threshold in the "middle" of the spike.

	time = np.arange(0,data.shape[0],1.0) #time array 
	time *= dt
	# print(time.shape)
	# print(mx_value,mn_value,th)

	event_indices = np.where(np.isclose(data+abs(mn_value), th,atol=tol))
	
	return time[event_indices]




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
def get_spikes(events,dt=0.001):
	# dt =0.001
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

#########  SINGLE INTERVALS


## on = 0; off = 1

##off1 - on1 
def get_burst_duration(data):
    return np.array([b-a for a,b in zip(data[:,0],data[:,1])])

##on2 - off1
def get_burst_interval(data):
	return np.array([a-b for a,b in zip(data[1:,0],data[:,1])])


##on2 - on1
def get_burst_period(data):
	return np.array([a-b for a,b in zip(data[1:,0],data[:,0])])


#########  PAIRED INTERVALS

def get_intervals(d1,d2):
	d1d2_interval = np.array([b-a for a,b in zip(d1[:,0],d2[:,0])])
	d1d2_delay = np.array([b-a for a,b in zip(d1[:,1],d2[:,0])])
	d2d1_interval = np.array([a-b for a,b in zip(d1[1:,0],d2[:-1,0])])
	d2d1_delay = np.array([a-b for a,b in zip(d1[1:,0],d2[:-1,1])])

	return [d1d2_interval,d1d2_delay],[d2d1_interval,d2d1_delay]


DUR = 0
IBI = 1
PER = 2

INTERVAL = 0
DELAY = 1

def analyse_pair(d1,d2,n1,n2,stats,index,plot=False):
	d1d2,d2d1 = get_intervals(d1,d2)

	print(n1,n2,"\t\t INTERVAL  \t\t   DELAY")
	print("\tMean: ",np.mean(d1d2[INTERVAL]),np.mean(d1d2[DELAY]))
	print("\tStd: ",np.std(d1d2[INTERVAL]),np.std(d1d2[DELAY]))

	print(n2,n1,"\t\t INTERVAL  \t\t   DELAY")
	print("\tMean: ",np.mean(d2d1[INTERVAL]),np.mean(d2d1[DELAY]))
	print("\tStd: ",np.std(d2d1[INTERVAL]),np.std(d2d1[DELAY]))


	# stats[str(index)+n1+n2] = [[np.mean(d1d2[INTERVAL]),np.std(d1d2[INTERVAL])],[np.mean(d1d2[DELAY]),np.std(d1d2[DELAY])]]
	# index +=1
	# stats[str(index)+n2+n1] = [[np.mean(d2d1[INTERVAL]),np.std(d2d1[INTERVAL])],[np.mean(d2d1[DELAY]),np.std(d2d1[DELAY])]]
	# index +=1

	stats[str(index[0])+n1+n2] = to_dict(d1d2,PAIR)
	index[0] +=1
	stats[str(index[0])+n2+n1] = to_dict(d2d1,PAIR)
	index[0] +=1

	return d1d2,d2d1


PAIR = 2
SINGLE = 1

def to_dict(data,type_):
	if(type_==PAIR):
		return {'Interval':data[:][INTERVAL],'Delay':data[:][DELAY]}
	elif(type_ == SINGLE):
		return {'Period':data[:][PER],'BD':data[:][DUR],'IBI':data[:][IBI]}
	else :
		return {}

def get_single_intervals(data):

	dur = get_burst_duration(data)

	ibi = get_burst_interval(data)

	period = get_burst_period(data)

	return dur,ibi,period

def analyse(data,neuron,stats,index,plot=False):

	n_intervals = get_single_intervals(data)


	print(neuron,"\t\t Duration  \t\t   IBI \t\t   Period")
	print("\tMean: ",np.mean(n_intervals[DUR]),np.mean(n_intervals[IBI]),np.mean(n_intervals[PER]))
	print("\tStd: ",np.std(n_intervals[DUR]),np.std(n_intervals[IBI]),np.std(n_intervals[PER]))
	print("\tStd: ",np.var(n_intervals[DUR]),np.var(n_intervals[IBI]),np.var(n_intervals[PER]))
	print("\tCV: ",np.std(n_intervals[DUR])/np.mean(n_intervals[DUR]),np.std(n_intervals[IBI])/np.mean(n_intervals[IBI]),np.std(n_intervals[PER])/np.mean(n_intervals[PER]))
	x = n_intervals[DUR][:-1]
	y=n_intervals[PER]
	cov =  np.matmul(x-np.mean(x), (y-np.mean(y)).T) / len(x)
	print("\tCovarianze BD and Period:",cov)
	print("\tR-squared expected:",cov**2/(np.var(n_intervals[DUR])*np.var(n_intervals[PER])))
	# if plot:
	# 	plot_hists([dur,ibi,period],neuron)

	# stats[neuron] = [[np.mean(dur),np.std(dur)],[np.mean(ibi),np.std(ibi)],[np.mean(period),np.std(period)]]
	stats[str(index[0])+neuron] = to_dict(n_intervals,SINGLE)
	index[0] += 1

	return n_intervals

#Equipares events length, fst is the reference
#3 must be same size

#input 3 events 
def fix_length(fst,snd,thr):
	fst,snd,thr = fix_init(fst,snd,thr)
	fst,snd,thr = fix_end(fst,snd,thr)
	return fst,snd,thr

def fix_init(fst,snd,thr):
	print(fst.shape,snd.shape,thr.shape)
	if (len(fst) != len(snd) or len(fst) != len(thr) or len(thr) != len(snd)):
		print(thr[0][0],fst[0][0])
		#thr on before fst off to solve overlaping
		while(thr[0][0]<fst[0][1]):
			thr = thr[1:]


		print("1",fst.shape,snd.shape,thr.shape)
		while(snd[0][0]<fst[0][0]):
			snd = snd[1:]

		print("2",fst.shape,snd.shape,thr.shape)

	return fst,snd,thr

def fix_end(fst,snd,thr):
	print("3",fst.shape,snd.shape,thr.shape)
	#print(snd[-1][0],thr[-1][0])
	#print(snd[-2][0],thr[-2][0])
	#print(snd[-3][0],thr[-3][0])
	while(len(thr)>len(snd) or len(thr)>len(fst)):
		thr = thr[:-1]
	if (len(fst) != len(snd) or len(fst) != len(thr) or len(thr) != len(snd)):
		while(snd[-1][0] > thr[-1][0]):
			snd = snd[:-1]

		print("4",fst.shape,snd.shape,thr.shape)
		while(fst[-1][0] > thr[-1][0]):
			fst = fst[:-1]

		print("5",fst.shape,snd.shape,thr.shape)


	return fst,snd,thr


# def fix_intervals(fst,snd,thr):
# 	for i,(n1,n2,n3) in enumerate(zip(fst,snd,thr)):
# 		if n3[0] < n1[0] or n2[0] < n1[0]:
# 			rm.append(i)
# 	fst = np.delete(fst,rm)






###?¿? = analyse

# def analyse_hists(data,neuron,plot=True):
# 	charac = analyse(data)

# 	return charac

