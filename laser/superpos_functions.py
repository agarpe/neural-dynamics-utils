import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
from random import randint
from colour import Color


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
	
def read_from_events(path,dt =0.1, max_cols = 300, delim="\t",dataview=False):
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


def set_plot_info(axes,labels,loc="best",xlabel="Time (ms)",ylabel="Voltage (mV)"):
	plt.legend(axes,labels,loc=loc)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)



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

	if(amplitude > 1): #Ignore artefacts
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

def parse_color(col):
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
			luminances = np.arange(1,0.2,-0.8/len(events))
			colors = list(col[0].range_to(col[1],len(events)))
		except: #when color is a single color
			fst_color,last_color = col,col
			col = col.hex_l

	return col,fst_color,last_color,colors



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
def plot_events(events,col,tit,width_ms=50,dt=0.1,df_log={},show_durations=False,error=False):
	if len(events)==0:
		print("Error: no events to plot")
		return plt.plot([]),plt.plot([]),plt.plot([])

	ax=0

	# if(isinstance(col,str)):
	# #set first and last spike colors
	# 	try:#fix in superpos scripts colors dict. Change name or change dict concept.
	# 		# colors = COLORS[col]
	# 		fst_color,last_color = COLORS[col] #when color is a tupple of strings

	# 	except:
	# 		fst_color,last_color=col,col
	# else: #when color is a tupple of Color
	# 	try:
	# 		fst_color = col[0].hex_l
	# 		last_color = col[1].hex_l
	# 		luminances = np.arange(1,0.2,-0.8/len(events))
	# 		colors = list(col[0].range_to(col[1],len(events)))
	# 	except: #when color is a single color
	# 		fst_color,last_color = col,col
	# 		col = col.hex_l

	col,fst_color,last_color,colors =  parse_color(col)

	# print(colors)


	count =[0]
	# ax_fst,=plt.plot([],[])
	# ax_last,=plt.plot([],[])
	# ax,=plt.plot([],[])
	ploted=0
	for spike_i in range(events.shape[0]):
		#remove possible nan values:
		spike = events[spike_i,:][~np.isnan(events[spike_i,:])]

		# prepare spike
		try:
			spike = center(spike,width_ms,dt) #center spike from max
			spike = no_drift(spike) #adjust drift
		except:
			print("skip ",spike_i)
			continue

		# get stat info
		df_log = get_spike_info(df_log,spike,dt,show_durations,spike_i,count)
		if(count[0] >20): #failed event
			break

		#Calculate time
		time = np.arange(0,spike.shape[0],1.0) #points to width_ms. 
		time *= dt

		# if(df_log['amplitude'][-1]>11):
		# 	print(spike_i)
		# 	break

		ax_fst,=plt.plot([],[],color=fst_color)
		ax_last,=plt.plot([],[],color=last_color)
		
		try:
			# col.luminance = luminances[spike_i%(len(events))]
			# color = col.hex_l
			color = colors[spike_i]
			color = color.hex_l
		except:
			color = col

		#TODO: fix failure when fst or last spikes are ignored
		#Plot first, last or general spike.
		if(spike_i==0):
			ax_fst,=plt.plot(time,spike,color=fst_color,linewidth=1.5)
		elif(spike_i==events.shape[0]-1):
			ax_last,=plt.plot(time,spike,color=last_color,linewidth=1.5)
		else:
			ax,=plt.plot(time,spike,color=color,linewidth=0.1)
		ploted+=1
	print(ploted)
			# ax_last,=plt.plot(time,spike,color=last_color,linewidth=1.5)
			# ax,=plt.plot(time,spike,linewidth=0.1) #darker effect ?
	plt.title(tit + str(ploted))
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


# Description: 
#	Detects the maximum value of the spike and takes width milliseconds to the left and
# 	width ms to the right, using dt to calculate the number of points necessaries. 
# Parameters:
#	spike voltage values
#	width_ms milliseconds to save at each side. 
# 	dt Data adquisition time
def center(spike,width_ms,dt=0.1):
	spike = spike[~np.isnan(spike)] 

	mx_index = np.argmax(spike) #index of maximum V value (spike)

	width_points = width_ms /dt #Number of points corresponding to the iteration
	
	ini = int(mx_index-width_points) #init as max point - number of points. 
	end = int(mx_index+width_points) #end as max point + number of points. 

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
	####

	return spike[ini:end]



# Description: 
# 	Recives spike voltage values and normalizes drift based on its minimum
# Parameters:
# 	spike voltage values

def no_drift(spike):
	if(spike.shape[0]!=0):
		spike = spike[~np.isnan(spike)] 
		mn = np.min(spike)
		if mn != 0:
			spike = spike-mn
	
	return spike


# Description: 
# 	Recives spike values and return the spike duration as a tuple of the time
# 	references of two of the values (in ms) matching a threshold in "the middle" of the spike
# Parameters:
# 	spike voltage values
# 	dt time rate
# 	tol difference tolerance (lower than 0.2 fails)
# Return:
#	(min_thres,max_thres)
def get_spike_duration(spike,dt,tol=0.2): 
	spike = spike[~np.isnan(spike)]
	mx_value = np.max(spike) #maximum V value (spike)
	mn_value = np.min(spike) #minimum V value (spike)

	th = (mx_value+mn_value)/2 #threshold in the "middle" of the spike.

	#Warning: with a lower tolerance value the threshold detection might fail
	duration_vals = np.where(np.isclose(spike, th,atol=tol))[0]

	if duration_vals.size ==0: #Safety comprobation
		return (0,0),th
	else:
		return (duration_vals[0]*dt,duration_vals[-1]*dt),th




# Description: 
# 	Recives spike values and return the amplitude value measured as the distance between
#	maximum and minimum voltage value.
# Parameters:
# 	spike voltage values
# 	dt time rate
# Return:
#	amplitude
def get_spike_amplitude(spike,dt):
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

def get_slope(spike,dt):
	spike = spike[~np.isnan(spike)] 
	mid_ps,th = get_spike_duration(spike,dt)
	indx1 = int(mid_ps[0]/dt) #From ms to point ref
	indx2 = int(mid_ps[1]/dt) #From ms to point ref

	slope1 = (spike[indx1]-spike[indx1-1])/dt 
	slope2 = (spike[indx2]-spike[indx2-1])/dt

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
