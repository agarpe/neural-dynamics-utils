
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
from random import randint



#TODO: 
# - "ignore artefacts, random value 5..."
def plot_events(events,col,tit,ms=50,dt=0.1,amplitude_log=0,show_amplitudes=False):
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

	for row_i in range(events.shape[0]):
		row = center(events[row_i,:],ms,dt) #center spike from max
		row = no_drift(row) #adjust drift

		if amplitude_log!=0:
			#Measure amplitudes:
			amplitudes,th = get_spike_amplitude(row,dt,tol=0.2)
			amp =  amplitudes[1]-amplitudes[0]
			if(amp > 5): #Ignore artefacts
				amplitude_log.append(amplitudes[1]-amplitudes[0])
			else:
				print("ignored value")

			if show_amplitudes:
				plt.plot(amplitudes,(th,th),'.',color='k') 

		# print(row.shape)
		time = np.arange(0,row.shape[0],1.0) #points to ms. 
		time *= dt

		# print(time.shape)
		if(row_i==0):
			ax_fst,=plt.plot(time,row,color=fst_color,linewidth=1.5)
		elif(row_i==events.shape[0]-1):
			ax_last,=plt.plot(time,row,color=last_color,linewidth=1.5)
		else:
			ax,=plt.plot(time,row,color=col,linewidth=0.1)	
			# ax,=plt.plot(time,row,linewidth=0.1) #darker effect ?
	plt.title(tit)

	return ax,ax_fst,ax_last



#Center spike from max
def center(spike,ms,dt=0.1):
	mx_index = np.argmax(spike) #index of maximum V value (spike)
	ms_points = ms /dt #Number of points corresponding to the iteration
	# ms_points = ms
	

	# print(mx_index)
	ini = int(mx_index-ms_points) #init as max point - number of points. 
	end = int(mx_index+ms_points) #end as max point + number of points. 

	###Beta func: Beware in models
	if mx_index!=0: #ignore artefacts
		#Adjust window when there are not enough points 
		if(ini < 0):
			app = np.full(abs(ini),spike[0]) 
			spike =np.insert(spike,0,app) #Add events at the begining
			return center(spike,ms,dt) #re-center
		if(end > spike.shape[0]):
			app = np.full(end-spike.shape[0],spike[-1]) #Add events at the end
			spike = np.insert(spike,spike.shape[0],app) #re-center
			return center(spike,ms,dt)

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
# 	Recives spike values and return the amplitude as a tuple of the time
# 	references of two of the values matching a threshold in "the middle" of the spike
# Parameters:
# 	spike voltage values
# 	dt time rate
# 	tol difference tolerance (lower than 0.2 fails)
# Return:
#	(min_thres,max_thres)
def get_spike_amplitude(spike,dt,tol=0.2): 
	mx_value = np.max(spike) #maximum V value (spike)
	mn_value = np.min(spike) #minimum V value (spike)

	th = (mx_value-mn_value)/2 #threshold in the "middle" of the spike.

	#Warning with a lower tolerance value the threshold detection might fail
	amplitude_vals = np.where(np.isclose(spike, th,atol=tol))[0]

	# print(th)
	# print(amplitude_vals)
	return (amplitude_vals[0]*dt,amplitude_vals[-1]*dt),th


