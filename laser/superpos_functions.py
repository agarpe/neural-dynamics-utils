import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
from random import randint
from colour import Color
from scipy.signal import argrelmax, find_peaks, peak_widths



def close_all_figs():
	import matplotlib.pyplot as plt
	figs = list(map(plt.figure, plt.get_fignums()))

	for fig in figs:
		# print(fig)
		plt.close(fig)
		
def create_dataframe(dicts,prefixes):
	if len(dicts) != len(prefixes):
		print("Error creating dataframe, dicts and prefixes with diffrent shapes")
		return {}

	dfs = []

	for d,p in zip(dicts,prefixes):
		df = pd.DataFrame.from_dict(d, orient='index')
		df = df.transpose()
		dfs.append(df.add_prefix(p))

	df = pd.concat(dfs,axis=1)

	return df
	
def read_from_events(path,dt =0.1, max_cols = 300, delim="\t",dataview=True):
	#Column names list generation to read files with distinct number of columns in each row. 
	#Indispensable when events obtained by threshold detection in DataView
	# dt =0.1
	# max_cols = 300 ##counting rows from file requires reading the whole file too long. 
	col_names = [i for i in range(0, int(max_cols/dt))]

	if dataview:
		#changes , by . as separator (for dataview)
		os.system("sed -i 's/\,/./g' "+path)

	#Each row contains Voltage values of the corresponding event.
	events =  pd.read_csv(path, delimiter = delim,skiprows=0,header=None,names=col_names)

	if events.shape[0]>10000:
		print(events.shape)
		return []
	return events


def set_plot_info(axes,labels,width,loc="upper left",xlabel="Time (ms)",ylabel="Voltage (mV)",xlim='default', ylim=None):
	# if labels != []:
	# 	plt.legend(axes,labels,loc=loc,prop={'size': 25})
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)

	# plt.xticks(np.arange(0, width*2, 5))
	# plt.yticks(np.arange(-100, 5, 5))

	if xlim == 'default':
		plt.xlim(0,width*2)
	elif xlim is not None:
		plt.xlim(xlim)
	if ylim is not None:
		plt.ylim(-100, 5)




def get_spike_info(df_log,spike,dt,show_durations,spike_i,error_count):
	if(df_log == {}): #if first spike
		df_log['duration']=[];df_log['amplitude']=[];
		df_log['slope_dep']=[];df_log['slope_rep']=[];
		df_log['slope_dep_max']=[];df_log['slope_rep_max']=[]

	if spike.shape[0]==0:
		return df_log

	#Measure durations:
	durations,th = get_spike_duration(spike,dt,tol=1)
	dur =  durations[1]-durations[0]

	if(dur > 1): #Ignore artefacts
		df_log['duration'].append(dur)
	else:
		print("ignored with index %d and duration value %f"%(spike_i,dur))
		error_count[0]+=1

	if show_durations:
		plt.plot(durations,(th,th),'.',color='k') 

	#Measure durations:
	amplitude = get_spike_amplitude(spike,dt)

	# if(amplitude > 0.5 and amplitude < 110): #Ignore artefacts
	if(amplitude > 20 and amplitude < 120): #Ignore artefacts
	# if(amplitude >= 80): #Ignore artefacts
		df_log['amplitude'].append(amplitude)
	else:
		print("ignored with index %d and amplitude value %f"%(spike_i,amplitude))
		error_count[0]+=1

	slope_inc,slope_dec = get_slope(spike,dt)
	df_log['slope_dep'].append(slope_inc)
	df_log['slope_rep'].append(slope_dec)	

	##########TODO: QUITAR
	slope_inc,slope_dec = get_slope_max(spike,dt)
	df_log['slope_dep_max'].append(slope_inc)
	df_log['slope_rep_max'].append(slope_dec)	

	return df_log	


COLORS = {'b':['cyan','darkblue'],'r':['coral','maroon'],'g':['lime','darkgreen']}

def parse_color(col,n_events=0):
	if col is None:
		return None, None, None, None
	colors=None
	if(isinstance(col,str)):
	#set first and last spike colors
		try:#fix in superpos scripts colors dict. Change name or change dict concept.
			# colors = COLORS[col]
			fst_color,last_color = COLORS[col] #when color is a tupple of strings

		except:
			fst_color,last_color=col,col
	else: #when color is a tupple of Color
		try:
			fst_color = col[0].hex_l
			last_color = col[1].hex_l
			luminances = np.arange(1,0.2,-0.8/n_events)
			colors = list(col[0].range_to(col[1],n_events))
		except: #when color is a single color
			fst_color,last_color = col,col
			col = col.hex_l

	return col,fst_color,last_color,colors

def align_spike(spike,width_ms_l,width_ms_r,dt,id_,mode='ini'):

	# prepare spike
	try:
	# spike = np.array(spike)
	# print(type(spike))
		spike = center(spike,width_ms_l, width_ms_r,dt) #center spike from max
		# spike = no_drift(spike,dt) #adjust drift
		spike = align_to(spike,mode, dt)
		return spike

	except:
		print("----skip ",id_)
		return []


# remove invalid spikes from waveforms array
# error in ms
def preprocess_spikes(spikes, refs,width_l, width_r=0, error=10):

	print("PREPROCESSING")
	spikes_copy = np.zeros(spikes.shape)

	ids = []
	print(spikes_copy.shape)
	print(refs.shape)
	for i,(event,ref) in enumerate(zip(spikes,refs)):
		# print(ref)
		if ref < error:
			spikes_copy[i,:] = event[:]
			ids.append(i)
		else: 
			print("Ignoring with stim distance: %f"%(ref))

	return spikes_copy, ids


# Description: 
#	Plots several spike events superpossed. When duration_log is a list, returns info 
# 	of the spikes duration.
# 	Before ploting the spike is centered from its maximum and width_ms to left and right.
# 	Drift is fixed normalizing to each spike minimum
# Parameters:
#	events array with spikes voltage values. Each row has all the voltage values from a spike. 
#	col plot color tuple of strings or Color.
#		
#	tit plot title
#	width_ms milliseconds to save at each side. 
# 	dt Data adquisition time
#	duration_log List where info from spikes duration is saved. Ignored when =0. 
#	show_durations when True detected durations are ploted. 
def plot_events(events,col,tit,width_ms_l=50,width_ms_r=50,dt=0.1,df_log={},show_durations=False,error=False,mode='ini',lw = 0.5):
	if len(events)==0:
		print("Error: no events to plot")
		return plt.plot([]),plt.plot([]),plt.plot([])

	ax=0
	col,fst_color,last_color,colors =  parse_color(col,len(events))


	count =[0]
	ax_fst,=plt.plot([],[])
	ax_last,=plt.plot([],[])
	ax,=plt.plot([],[])
	ploted=0
	for spike_i in range(events.shape[0]):
	# for spike_i in range(1):
		#remove possible nan values:
		spike = events[spike_i,:][~np.isnan(events[spike_i,:])]

		spike = align_spike(spike, width_ms_l, width_ms_r,dt,spike_i,mode)
		if spike == []:
			continue

		# get stat info
		prev = count[0]
		df_log = get_spike_info(df_log,spike,dt,show_durations,spike_i,count)
		if(count[0]-prev >0): #skip events that have failed.
			continue

		#Calculate time
		time = np.arange(0,spike.shape[0],1.0) #points to width_ms. 
		time *= dt

		ax_fst,=plt.plot([],[],color=fst_color)
		ax_last,=plt.plot([],[],color=last_color)
		
		try:
			color = colors[spike_i]
			color = color.hex_l
		except:
			color = col

		#TODO: fix failure when fst or last spikes are ignored 	
		#Plot first, last or general spike.
		# if(spike_i==0):
		# 	ax_fst,=plt.plot(time,spike,color=fst_color,linewidth=1.5)
		# elif(spike_i==events.shape[0]-1):
		# 	ax_last,=plt.plot(time,spike,color=last_color,linewidth=1.5)
		# else:
		ax,=plt.plot(time,spike,color=color,linewidth=lw)
		ploted+=1
		# plt.show()

	plt.title(tit + " " +str(ploted))

	#TODO: REVIEW the df reduction ?¿?¿
	# print(len(events),ploted)
	if count[0] >0:
		# print([len(df_log[x]) for x in df_log if isinstance(df_log[x], list)])
		if not error: #In case there is no error allowance the dict is reduced.
			df_log['amplitude']=df_log['amplitude'][:-count[0]]
			df_log['slope_dep']=df_log['slope_dep'][:-count[0]]
			df_log['slope_rep']=df_log['slope_rep'][:-count[0]]
		print(count,"\"spikes\" ignored")

	return ax,ax_fst,ax_last

def simple_plot(events,col,tit,width_ms=50,dt=0.1,df_log={},show_durations=False,error=False):
	for spike_i in range(events.shape[0]):
		#remove possible nan values:
		spike = events[spike_i,:][~np.isnan(events[spike_i,:])]

		spike = no_drift(spike)

		#Calculate time
		time = np.arange(0,spike.shape[0],1.0) #points to width_ms. 
		time *= dt

		parse_color(col)
		try:
			# col.luminance = luminances[spike_i%(len(events))]
			# color = col.hex_l
			color = colors[spike_i]
			color = color.hex_l
		except:
			color = col

		ax,=plt.plot(time,spike,color=color,linewidth=0.1)
		plt.title(tit)

	return ax,ax,ax

def burst_plot(events,col,tit,width_ms=50,dt=0.1,df_log={},show_durations=False,error=False):

	peaks = np.max(events,axis=1)
	val = np.mean(peaks[~np.isnan(peaks)])
	print(val, peaks)

	for spike_i in range(events.shape[0]):
		#remove possible nan values:
		spike = events[spike_i,:][~np.isnan(events[spike_i,:])]

		spike = no_drift(spike)


		# spike = align_to(spike, mode='ini')
		# spike = align_to(spike,val)

		#Calculate time
		time = np.arange(0,spike.shape[0],1.0) #points to width_ms. 
		time *= dt

		parse_color(col)
		try:
			# col.luminance = luminances[spike_i%(len(events))]
			# color = col.hex_l
			color = colors[spike_i]
			color = color.hex_l
		except:
			color = col

		ax,=plt.plot(time,spike,color=color,linewidth=0.1)
		plt.title(tit)

	return ax,ax,ax

def plot_events_mean(events,col,tit,width_ms_l=50,width_ms_r=50,dt=0.1,df_log={},show_durations=False,error=False,mode='peak'):
	if len(events)==0:
		print("Error: no events to plot")
		return plt.plot([]),plt.plot([]),plt.plot([])

	ax=0
	col,fst_color,last_color,colors =  parse_color(col,len(events))

	count =[0]
	ploted=0
	aligned_spikes = []
	for spike_i in range(events.shape[0]):
		#remove possible nan values:
		spike = events[spike_i,:][~np.isnan(events[spike_i,:])]

		spike = align_spike(spike, width_ms_l, width_ms_r,dt,spike_i,mode)
		if spike == []:
			continue

		# get stat info
		prev = count[0]
		df_log = get_spike_info(df_log,spike,dt,show_durations,spike_i,count)
		if(count[0]-prev >0): #skip events that have failed.
			continue
		if spike != []:
			ploted+=1
			aligned_spikes.append(list(spike))

	mean_spike = np.average(aligned_spikes, axis=0)

	#Calculate time
	time = np.arange(0,mean_spike.shape[0],1.0) #points to width_ms. 
	time *= dt

	try:
		color = colors[spike_i]
		color = color.hex_l
	except:
		color = col


	ax_fst,=plt.plot([],[])
	ax_last,=plt.plot([],[])
	ax,=plt.plot(time,mean_spike,color=color)

	plt.title(tit + " N=" +str(ploted))

	print(len(events),ploted)
	if count[0] >0:
		# print([len(df_log[x]) for x in df_log if isinstance(df_log[x], list)])
		if not error: #In case there is no error allowance the dict is reduced.
			df_log['amplitude']=df_log['amplitude'][:-count[0]]
			df_log['slope_dep']=df_log['slope_dep'][:-count[0]]
			df_log['slope_rep']=df_log['slope_rep'][:-count[0]]
		print(count,"\"spikes\" ignored")

	return ax,ax_fst,ax_last

# Description: 
#	Detects the maximum value of the spike and takes width milliseconds to the left and
# 	width ms to the right, using dt to calculate the number of points necessaries. 
# Parameters:
#	spike voltage values
#	width_ms milliseconds to save at each side. 
# 	dt Data adquisition time
def center(spike,width_ms_l, width_ms_r,dt=0.1):
	spike = spike[~np.isnan(spike)] 

	mx_index = np.argmax(spike) #index of maximum V value (spike)

	width_points_l = width_ms_l /dt #Number of points corresponding to the iteration
	width_points_r = width_ms_r /dt #Number of points corresponding to the iteration
	
	ini = int(mx_index-width_points_l) #init as max point - number of points. 
	end = int(mx_index+width_points_r) #end as max point + number of points. 

	# ###Beta func: Beware in models
	# if mx_index!=0: #ignore artefacts
	# 	#Adjust window when there are not enough points 
	# 	if(ini < 0):
	# 		app = np.full(abs(ini),spike[0]) 
	# 		spike =np.insert(spike,0,app) #Add events at the begining
	# 		return center(spike,width_ms,dt) #re-center
	# 	if(end > spike.shape[0]):
	# 		app = np.full(end-spike.shape[0],spike[-1]) #Add events at the end
	# 		spike = np.insert(spike,spike.shape[0],app) #re-center
	# 		return center(spike,width_ms,dt)
	# ###
	###Beta func: V2 --> 
	if mx_index!=0: #ignore artefacts
		#Adjust window when there are not enough points 
		if(ini < 0):
			return [] #re-center
		if(end > spike.shape[0]):
			return []
	###

	return spike[ini:end]



# Description: 
# 	Recives spike voltage values and normalizes drift based on its minimum
# Parameters:
# 	spike voltage values

def no_drift(spike,mode='first_min',dt=0.1):
	print(np.array(spike))
	if(spike.shape[0]!=0):
		spike = spike[~np.isnan(spike)] 
		mn = np.min(spike)
		if mn != 0:
			spike = spike-mn
	
	return spike

# Description: 
# 	Recives spike voltage values and normalizes drift based on its minimum
# Parameters:
# 	spike voltage values

# sec_wind window in ms to measure the slope between two points

def align_to(spike,mode='peak',dt=0.1,sec_wind=2.0):
	# mode = 'peak'
	spike = np.array(spike)
	if(spike.shape[0]!=0):
		if mode == 'min':
			indx = np.argmin(spike)
			mn = np.min(spike)
		elif mode == 'peak':
			indx = np.argmax(spike)
			mn = np.max(spike)
		elif mode == 'ini':
			indx = 0
			mn = spike[0]
		elif mode == 'first_min':
			sec_wind = int(sec_wind/dt)
			slopes = [ (s1-s2)/dt for s1,s2  in zip(spike[:spike.shape[0]//2-1],spike[sec_wind:spike.shape[0]//2-1])]
			# # indx = np.argmin(slopes)
			
			slopes = np.array(slopes)
			# slopes = abs(slopes[np.where(slopes<0)])
			# slopes = slopes[np.where(slopes>0)]
			# diffs = slopes[1:]-slopes[:-1]
			# # diffs = slopes[:-1]-slopes[1:]
			# # print(np.mean(diffs),max(diffs),min(diffs))
			# th = max(abs(diffs))
			# indx = np.where(abs(diffs)==th)[0][0]
			# # diffs = diffs[np.where(diffs <0)]
			# indx = np.argmax(diffs)
			# # indx = np.argmin(diffs)
			# # indx = np.where(spike>np.min(spike)+10)[0][0]
			indx = np.argmax(slopes)
			# print(slopes)
			mn = spike[indx]
			time = np.arange(0,spike.shape[0]//2,1.0) #points to width_ms. 
			time *= dt
			time=time[indx]
			# mn = spike[0]

		elif mode == 'first_max':
			indx = argrelmax(spike)[0][0]
			# plt.plot(fst_max,spike[fst_max])
			# print(fst_max)
			mn = spike[indx]
		
		elif type(mode) is not str:
			mn = mode
			indx = 10
			
		else:
			print("fail")
			print(type(mode))
		# mn = np.min(spike)
		if mn != 0:
			spike = spike-mn
			time = indx*dt

			# plt.plot(time,mn,'.',color='k') 
			# plt.plot(np.ones(spike[np.where(slopes>0)].shape)*dt,spike[np.where(slopes>0)],'.',color='k') 
	
	return spike


# # Description: 
# # 	Recives spike values and return the spike duration as a tuple of the time
# # 	references of two of the values (in ms) matching a threshold in "the middle" of the spike
# # Parameters:
# # 	spike voltage values
# # 	dt time rate
# # 	tol difference tolerance im mV (lower than 2 fails)
# # Return:
# #	(min_thres,max_thres)
# def get_spike_duration(spike,dt,tol=2, thres_val=0.5): 
# 	spike = spike[~np.isnan(spike)]

# 	mx_value = np.max(spike) #maximum V value (spike)
# 	mn_value = np.min(spike) #minimum V value (spike)

# 	#TODO: check with neurons with pos and neg val...
# 	th = (mx_value+mn_value)*thres_val #threshold in the "middle" of the spike.

# 	#Warning: with a lower tolerance value the threshold detection might fail
# 	duration_vals = np.where(np.isclose(spike, th,atol=tol))[0]
# 	#TODO: tol with 0.1??? 

# 	# plt.plot(duration_vals,(th,th),'.',markersize=20,color='r')

# 	if duration_vals.size ==0: #Safety comprobation
# 		return (0,0),th
# 	else:
# 		return (duration_vals[0]*dt,duration_vals[-1]*dt),th

# Description: 
# 	Recives spike values and return the spike duration as a tuple of the time
# 	references of two of the values (in ms) matching a threshold in "the middle" of the spike
# Parameters:
# 	spike voltage values
# 	dt time rate
# 	tol difference tolerance im mV (lower than 2 fails) --> DEPRECATED
# Return:
#	(min_thres,max_thres)
def get_spike_duration(spike,dt,tol=2, thres_val=0.5, max_dur = 5, v_scale=1, plot=False): 
	spike = spike[~np.isnan(spike)]

	peaks, properties = find_peaks(spike, prominence=1*v_scale, width=20)
	print(peaks)

	#Warning: this may fail for not centered waveforms	
	if len(peaks)>1: # in case spike has several peaks, gets mid one.
		mid_peak = np.isclose(len(spike)//2, peaks, atol=10)
		peaks = peaks[mid_peak]
	
	print(peaks)
	results_half = peak_widths(spike, peaks, rel_height=thres_val)

	print(results_half)
	x=spike
	if plot:
		plt.figure()
		plt.plot(x)
		plt.hlines(results_half[1][0],results_half[2][0]*dt, results_half[3][0]*dt, color="C2")
	plt.show()
	try: 

		duration_vals = np.array([results_half[2][0], results_half[3][0]])
		th = results_half[1]
	except Exception as e:
		# print(e.args)

		# x=spike
		# plt.figure()
		# plt.plot(x)
		# plt.plot(peaks, x[peaks], "x")
		# plt.hlines(*results_half[1:], color="C2")
		# plt.show()
		duration_vals = np.array([])
		th = 0

	# plt.plot(duration_vals,(th,th),'.',markersize=20,color='r')
	

	if duration_vals.size == 0: #Safety comprobation
		return (0,0),th
	else:
		return (duration_vals[0]*dt,duration_vals[-1]*dt),th

def get_spike_duration_value(waveform, dt, plot=False, v_scale=1):
    dur_refs,th = get_spike_duration(waveform, dt, plot=plot, v_scale=v_scale)
    duration = dur_refs[1]-dur_refs[0]
    return duration


# Description: 
# 	Recives spike values and return the amplitude value measured as the distance between
#	maximum and minimum voltage value.
# Parameters:
# 	spike voltage values
# 	dt time rate
# Return:
#	amplitude
def get_spike_amplitude(spike,dt, v_scale=1, thres_val=None):
	spike = spike[~np.isnan(spike)] 
	mx_value = np.max(spike) #maximum V value (spike)
	mn_value = np.min(spike) #minimum V value (spike)

	return mx_value-mn_value



# Description: 
# 	Recives spike values and return the increasing and decreasing slope values at the 
#	two points matching a threshold in "the middle" of the spike
#	maximum and minimum voltage value.
# Parameters:
# 	spike voltage values
# 	dt time rate
# Return:
#	amplitude

# def get_slope(spike,dt):
# 	spike = spike[~np.isnan(spike)] 
# 	mid_ps,th = get_spike_duration(spike,dt)
# 	indx1 = int(mid_ps[0]/dt) #From ms to point ref
# 	indx2 = int(mid_ps[1]/dt) #From ms to point ref

# 	slope1 = (spike[indx1]-spike[indx1-1])/dt 
# 	slope2 = (spike[indx2]-spike[indx2-1])/dt

# 	return (slope1,slope2)

# # v2 slopes from mid to max
# def get_slope(spike,dt):
# 	spike = spike[~np.isnan(spike)] 
# 	mid_ps,th = get_spike_duration(spike,dt)
# 	mx_value = np.max(spike) #maximum V value (spike)
# 	t_max = np.argmax(spike) #maximum V value (spike)

# 	t1 = mid_ps[0]-2
# 	t2 = mid_ps[1]+2

# 	indx1 = int(t1/dt) #From ms to point ref
# 	indx2 = int(t2/dt) #From ms to point ref

# 	time = np.arange(spike.size) *dt
# 	plt.plot(time,spike,color='k')
# 	plt.plot([t1,t_max*dt], [spike[indx1],mx_value],color='b')
# 	plt.plot([t2,t_max*dt], [spike[indx2],mx_value],color='b')
# 	# plt.plot(t_max*dt, mx_value, '|', markersize=100)
# 	# plt.plot(t1, spike[indx1], '|', markersize=100)
# 	# plt.plot(t2, spike[indx2], '|', markersize=100)
# 	# plt.show()

# 	slope1 = (spike[indx1]-mx_value)/ (t1 - (t_max*dt))
# 	slope2 = (mx_value-spike[indx2])/((t_max*dt) - t2)

# 	return (slope1,slope2)

# v3 a few points in the middle
# spike: v values
# dt: data time step
# n_points: number of points around position to calculate slope
# slope_position: where to calculate slope: defalult value, mid of spike.
def get_slope(spike,dt,n_points=10, slope_position=0.5, plot=False, v_scale=1):
	spike = spike[~np.isnan(spike)] 
	mid_ps,th = get_spike_duration(spike,dt,thres_val=slope_position, v_scale=v_scale)
	indx1 = int(mid_ps[0]/dt) #From ms to point ref
	indx2 = int(mid_ps[1]/dt) #From ms to point ref

	scale = int(0.1/dt) #TODO: fix repol points and mid+ in ms not points. 
	n_points *= scale

	# n_points = int(n_ms /dt)
	n_ms = n_points*dt

	slope1 = (spike[indx1+n_points]-spike[indx1-n_points])/(n_ms*2) 
	slope2 = (spike[indx2+n_points]-spike[indx2-n_points])/(n_ms*2)


	#plot to test
	t1 = mid_ps[0]
	t2 = mid_ps[1]


	time = np.arange(spike.size) *dt

	# if slope1 < 7:
		# print(slope1, slope2)

	if plot:
		plt.plot(time,spike,color='k',alpha=0.2)
		plt.plot([t1-n_ms,t1+n_ms], [spike[indx1-n_points],spike[indx1+n_points]],color='b',linewidth=1.3)
		plt.plot([t2-n_ms,t2+n_ms], [spike[indx2-n_points],spike[indx2+n_points]],color='b',linewidth=1.3)
	# plt.show()

	# print(slope1, slope2)
	# print(n_ms)
	# print(indx1, indx2)
	# print(spike[indx1+n_points],spike[indx2+n_points])
	# exit()

	return (slope1,slope2)

def get_slope2(spike,dt,n_points=5, slope_position=0.8, repol_points=60, plot=False):
	spike = spike[~np.isnan(spike)] 
	mid_ps,th = get_spike_duration(spike,dt,thres_val=slope_position)
	indx1 = int(mid_ps[0]/dt) #From ms to point ref
	indx2 = int(mid_ps[1]/dt) #From ms to point ref
	mx_value = np.max(spike) #maximum V value (spike)
	t_max = np.argmax(spike) #maximum V value (spike)	

	scale = int(0.1/dt) #TODO: fix repol points and mid+ in ms not points. 
	repol_points *= scale
	n_points *= scale


	t2 = mid_ps[1]+ 2
	indx2 = int(t2/dt) #From ms to point ref

	# # n_points = int(n_ms /dt)
	# half1 = spike[:t_max]
	# half2 = spike[t_max:]
	# # print(n_points)
	# n_points = int((n_points/100)*len(half1[np.argmin(half1):]))
	# print(n_points)

	n_ms = n_points*dt

	slope1 = (spike[indx1+n_points]-spike[indx1-n_points])/(n_ms*2) 
	# slope2 = (spike[indx2+n_points]-spike[indx2-n_points])/(n_ms*2)

	if (t_max+repol_points) > len(spike):
		repol_points = len(spike)-t_max-1

	slope2 = (mx_value-spike[t_max+repol_points])/((t_max*dt) - (t_max+repol_points)*dt)

	#plot to test
	t1 = mid_ps[0]
	t2 = mid_ps[1]

	time = np.arange(spike.size) *dt

	if plot:
		# plt.plot(time,spike,color='k',alpha=0.2)
		plt.plot([t1-n_ms,t1+n_ms], [spike[indx1-n_points],spike[indx1+n_points]],color='b',linewidth=1.3)
		plt.plot([t_max*dt,(t_max+repol_points)*dt], [mx_value,spike[t_max+repol_points]],color='b',linewidth=1.3)
	# plt.show()
	# if slope1 < 7:
		# print(slope1, slope2)

	# plt.plot(time,spike,color='k',alpha=0.2)
	# plt.plot([t1-n_ms,t1+n_ms], [spike[indx1-n_points],spike[indx1+n_points]],color='b',linewidth=1.3)
	# plt.plot([t2-n_ms,t2+n_ms], [spike[indx2-n_points],spike[indx2+n_points]],color='b',linewidth=1.3)
	# plt.show()

	# print(slope1, slope2)
	# print(n_ms)
	# print(indx1, indx2)
	# print(spike[indx1+n_points],spike[indx2+n_points])
	# exit()

	return (slope1,slope2)



# Description: 
# 	Recives spike values and return the increasing and decreasing slope values at the 
#	two points matching a threshold in "the middle" of the spike
#	maximum and minimum voltage value.
# Parameters:
# 	spike voltage values
# 	dt time rate
# Return:
#	amplitude

def get_slope_max(spike,dt):
	spike = spike[~np.isnan(spike)] 
	mid_ps,th = get_spike_duration(spike,dt)
	mx_value = np.max(spike) #maximum V value (spike)

	indx1 = int(mid_ps[0]/dt) #From ms to point ref
	indx2 = int(mid_ps[1]/dt) #From ms to point ref

	slope1 = (spike[indx1]-mx_value)/dt 
	slope2 = (mx_value-spike[indx2-1])/dt

	return (slope1,slope2)


def normalize_by(df, type, metrics=['duration','depolarization slope','repolarization slope','amplitude']):
    control_means = df[df['type']==type].groupby('file').mean()
    control_means = control_means[metrics]

    control_stds = df[df['type']==type].groupby('file').std()
    control_stds = control_stds[metrics]
    # print(control_means)

    for group in control_means.index:
        # df.loc[df['file']==group,metrics] = df.loc[df['file']==group,metrics].apply(lambda x: abs(x)/abs(control_means.loc[group]), axis=1)
        df.loc[df['file']==group, metrics] = df.loc[df['file']==group, metrics].apply(lambda x: x/control_means.loc[group], axis=1)
        # df.loc[df['file']==group,metrics] = df.loc[df['file']==group, metrics].apply(lambda x: (abs(x)-abs(control_means.loc[group]))/control_stds.loc[group], axis=1)

    return df



def get_sample(df, n_spikes):
    if n_spikes < 0:
        return df, pd.DataFrame(columns=df.columns)
    discard =df.groupby(['file', 'type']).filter(lambda x: len(x) < n_spikes).reset_index(drop=True)
    df = df.groupby(['file', 'type']).filter(lambda x: len(x) >= n_spikes).reset_index(drop=True)

    for file in discard.groupby('file').groups.keys():
        df.drop(df.index[df['file']==file], inplace=True)

    df = df.groupby(['file', 'type']).apply(pd.DataFrame.sample,n=n_spikes).reset_index(drop=True)

    return df, discard


def clean_spikes(df, metrics, z_error=0.1):
	# Clean spikes with different shapes
	org_std = df['duration'].std()

	from scipy.stats import zscore
	for col in metrics:
	    df[col+'_std'] = df.groupby(['file','type'])[[col]].transform('std')
	    df[col+'_z'] = df.groupby(['file','type'])[[col]].transform(zscore)

	print("Number of spikes discarded:", df.groupby(['file','type']).count())

	# z_error = 0.1

	df_outliers = df[df.index.isin(df.loc[(df['duration_z'] >= -z_error) & (df['duration_std'] >= z_error)].index)].copy()
	df = df[df.index.isin(df.loc[(df['duration_z'] < -z_error) | (df['duration_std'] < z_error)].index)].copy()

	print(df_outliers.file)
	print(df.file)

	print("Number of spikes discarded:", df_outliers.groupby(['file','type']).count())
	return df

def get_burst_duration_value(
	burst_waveform, dt, slope_threshold=0.1, use_threshold=True,
	thres_val=0.5, window=20, v_scale=1, plot=False
	):
	ini, end = get_burst_duration(burst_waveform, dt, slope_threshold,
	thres_val, window, v_scale, plot)
	return end-ini

def get_burst_duration(
	burst_waveform, dt, slope_threshold=0.1,
	thres_val=0.5, window=20, v_scale=1, plot=False
	):
	"""
	Calculates burst duration from a burst waveform segment using slope or threshold.

	Parameters:
	- burst_waveform: 1D array of the burst waveform
	- dt: time step (e.g., in ms)
	- slope_threshold: threshold for dV/dt when not using amplitude
	- use_threshold: if True, use amplitude threshold instead of slope
	- thres_val: amplitude threshold as fraction of peak (only if use_threshold=True)
	- window: number of points to search before/after peaks
	- v_scale: peak prominence factor for find_peaks
	- plot: show waveform and markers

	Returns:
	- (start_time, end_time): times in same units as dt
	"""

	burst_waveform = burst_waveform[~np.isnan(burst_waveform)]

	# Commented version because slows down execution
	# # Find all spike peaks in the burst
	# peaks, _ = find_peaks(burst_waveform, prominence=1*v_scale)
	# if len(peaks) < 1:
	# 	return (0, 0)

	# first_peak = peaks[0]
	# last_peak = peaks[-1]

	# Define search window around the burst
	# search_start = max(0, first_peak - window)
	# search_end = min(len(burst_waveform), last_peak + window)
	# segment = burst_waveform[search_start:search_end]
	
	segment = burst_waveform

	peak_val = np.max(segment)
	base_val = np.min(segment)
	threshold_val = base_val + thres_val * (peak_val - base_val)

	# Find first crossing above threshold
	depol_idx = np.argmax(segment > threshold_val)
	# Find last crossing above threshold
	repol_idx = len(segment) - np.argmax(segment[::-1] > threshold_val) - 1

	# # Convert back to full waveform indices
	# depol_idx += search_start
	# repol_idx += search_start

	# Convert to time
	start_time = depol_idx * dt
	end_time = repol_idx * dt

	if plot:
		t = np.arange(len(burst_waveform)) * dt
		plt.figure(figsize=(10, 4))
		plt.plot(t, burst_waveform, label='Burst waveform')
		# plt.plot(peaks * dt, burst_waveform[peaks], 'rx', label='Spikes')
		# plt.axvline(start_time, color='g', linestyle='--', label='Depol Start')
		# plt.axvline(end_time, color='r', linestyle='--', label='Repol End')
		# plt.axvspan(first_peak * dt, last_peak * dt, color='orange', alpha=0.2, label='Peak Range')
		plt.hlines(threshold_val, t[0], t[-1], color='purple', linestyle=':', label='Threshold')
		plt.legend()
		plt.title("Burst Duration via Threshold")
		plt.xlabel("Time")
		plt.ylabel("Amplitude")
		plt.tight_layout()
		plt.show()

	return start_time, end_time

def get_burst_slope(
    waveform, dt, n_points=10,
    slope_threshold=0.1,
    thres_val=0.5, v_scale=1, window=20, plot=False
):
	"""
	Compute depolarization and repolarization slopes from a waveform segment
	using start/end timepoints from get_burst_duration_from_waveform().
	thres_val: slope_position
	Returns:
	- (slope1, slope2): depolarization and repolarization slopes
	"""
	# 1. Get time of depolarization and repolarization
	start_time, end_time = get_burst_duration(
		burst_waveform=waveform,
		dt=dt,
		slope_threshold=slope_threshold,
		thres_val=thres_val,
		v_scale=v_scale,
		window=window,
		plot=False  # Don't double-plot here
	)

	# 2. Convert time to indices
	idx1 = int(start_time / dt)
	idx2 = int(end_time / dt)

	half_window = int(n_points)
	n_ms = dt * 2 * half_window

	if (idx1 - half_window < 0 or idx1 + half_window >= len(waveform) or
		idx2 - half_window < 0 or idx2 + half_window >= len(waveform)):
		return 0, 0


	# 3. Compute slopes
	slope1 = (waveform[idx1 + half_window] - waveform[idx1 - half_window]) / n_ms
	slope2 = (waveform[idx2 + half_window] - waveform[idx2 - half_window]) / n_ms

	# 4. Plot if needed
	if plot:
		t = np.arange(len(waveform)) * dt
		plt.figure(figsize=(10, 4))
		plt.plot(t, waveform, label='Waveform', alpha=0.6)
		plt.axvline(start_time, color='g', linestyle='--', label='Depol')
		plt.axvline(end_time, color='r', linestyle='--', label='Repol')
		plt.plot(
			[dt * (idx1 - half_window), dt * (idx1 + half_window)],
			[waveform[idx1 - half_window], waveform[idx1 + half_window]],
			'b', linewidth=2, label='Slope 1'
		)
		plt.plot(
			[dt * (idx2 - half_window), dt * (idx2 + half_window)],
			[waveform[idx2 - half_window], waveform[idx2 + half_window]],
			'b', linewidth=2, label='Slope 2'
		)
		plt.title("Waveform Slopes via get_burst_duration")
		plt.xlabel("Time (ms)")
		plt.ylabel("Amplitude")
		plt.legend()
		plt.tight_layout()
		plt.show()

	return slope1, slope2
