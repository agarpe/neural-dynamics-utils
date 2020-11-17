import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
from random import randint

def read_from_events(path,dt =0.1, max_cols = 300, delim="\t"):
	#Column names list generation to read files with distinct number of columns in each row. 
	#Indispensable when events obtained by threshold detection in DataView
	# dt =0.1
	# max_cols = 300 ##counting rows from file requires reading the whole file too long. 
	col_names = [i for i in range(0, int(max_cols/dt))]

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


# Description: 
#	Plots several spike events superpossed. When duration_log is a list, returns info 
# 	of the spikes duration.
# 	Before ploting the spike is centered from its maximum and width_ms to left and right.
# 	Drift is fixed normalizing to each spike minimum
# Parameters:
#	events array with spikes voltage values. Each row has all the voltage values from a spike. 
#	col plot color
#	tit plot title
#	width_ms milliseconds to save at each side. 
# 	dt Data adquisition time
#	duration_log List where info from spikes duration is saved. Ignored when =0. 
#	show_durations when True detected durations are ploted. 
def plot_events(events,col,tit,width_ms=50,dt=0.1,duration_log=0,amplitude_log=0,show_durations=False):
	ax=0
	if(col=='b'):
		fst_color = 'cyan'
		last_color = 'darkblue'
	elif(col=='r'): 
		fst_color = 'coral'
		last_color = 'maroon'
	elif(col=='g'):
		fst_color = 'lime'
		last_color = 'darkgreen'
	else:
		fst_color = col
		last_color = col

	count =0
	for spike_i in range(events.shape[0]):
		#remove possible nan values:
		spike = events[spike_i,:][~np.isnan(events[spike_i,:])]

		spike = center(spike,width_ms,dt) #center spike from max
		spike = no_drift(spike) #adjust drift

		if duration_log!=0 and spike.shape[0]!=0:
			#Measure durations:
			durations,th = get_spike_duration(spike,dt,tol=0.2)

			dur =  durations[1]-durations[0]

			if(dur > 1): #Ignore artefacts
				duration_log.append(dur)
			else:
				print("ignored duration value",spike_i)
				count+=1
				if(count >100):
					break

			if show_durations:
				plt.plot(durations,(th,th),'.',color='k') 

		if amplitude_log!=0 and spike.shape[0]!=0:
			#Measure durations:
			amplitude = get_spike_amplitude(spike,dt)

			if(amplitude > 1): #Ignore artefacts
				amplitude_log.append(amplitude)
			else:
				print("ignored amplitude value",spike_i)
				count+=1

		# print(spike.shape)
		#Calculate time
		time = np.arange(0,spike.shape[0],1.0) #points to width_ms. 
		time *= dt

		# print(time.shape)
		#Plot first, last or general spike.
		if(spike_i==0):
			ax_fst,=plt.plot(time,spike,color=fst_color,linewidth=1.5)
		elif(spike_i==events.shape[0]-1):
			ax_last,=plt.plot(time,spike,color=last_color,linewidth=1.5)
		else:
			ax,=plt.plot(time,spike,color=col,linewidth=0.1)	
			# ax,=plt.plot(time,spike,linewidth=0.1) #darker effect ?
	plt.title(tit)
	if count >0:
		print(count,"\"spikes\" ignored")
	return ax,ax_fst,ax_last



# Description: 
#	Detects the maximum value of the spike and takes width milliseconds to the left and
# 	width ms to the right, using dt to calculate the number of points necessaries. 
# Parameters:
#	spike voltage values
#	width_ms milliseconds to save at each side. 
# 	dt Data adquisition time
def center(spike,width_ms,dt=0.1):

	mx_index = np.argmax(spike) #index of maximum V value (spike)

	width_points = width_ms /dt #Number of points corresponding to the iteration
	
	ini = int(mx_index-width_points) #init as max point - number of points. 
	end = int(mx_index+width_points) #end as max point + number of points. 

	###Beta func: Beware in models
	if mx_index!=0: #ignore artefacts
		#Adjust window when there are not enough points 
		if(ini < 0):
			app = np.full(abs(ini),spike[0]) 
			spike =np.insert(spike,0,app) #Add events at the begining
			return center(spike,width_ms,dt) #re-center
		if(end > spike.shape[0]):
			app = np.full(end-spike.shape[0],spike[-1]) #Add events at the end
			spike = np.insert(spike,spike.shape[0],app) #re-center
			return center(spike,width_ms,dt)
	####

	return spike[ini:end]



# Description: 
# 	Recives spike voltage values and normalizes drift based on its minimum
# Parameters:
# 	spike voltage values

def no_drift(spike):
	if(spike.shape[0]!=0):
		mn = np.min(spike)
		if mn != 0:
			spike = spike-mn
	
	return spike


# Description: 
# 	Recives spike values and return the spike duration as a tuple of the time
# 	references of two of the values matching a threshold in "the middle" of the spike
# Parameters:
# 	spike voltage values
# 	dt time rate
# 	tol difference tolerance (lower than 0.2 fails)
# Return:
#	(min_thres,max_thres)
def get_spike_duration(spike,dt,tol=0.2): 
	mx_value = np.max(spike) #maximum V value (spike)
	mn_value = np.min(spike) #minimum V value (spike)

	th = (mx_value-mn_value)/2 #threshold in the "middle" of the spike.

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
	mx_value = np.max(spike) #maximum V value (spike)
	mn_value = np.min(spike) #minimum V value (spike)

	return mx_value-mn_value



def get_slope(spike,dt):
	mid_ps = get_spike_duration(spike,dt)
	indx1 = mid_ps[0]/dt
	indx2 = mid_ps[1]/dt 

	slope1 = (spike[indx1]-spike[indx1-1])/dt
	slope2 = (spike[indx2]-spike[indx2-1])/dt

	return slope1,slope2

