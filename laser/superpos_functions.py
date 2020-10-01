
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
from random import randint

def plot_events(events,col,tit,ms=50,dt=0.1):
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
		# ax,=plt.plot(row,color=col,linewidth=0.1)
			ax,=plt.plot(time,row,linewidth=0.1)
	plt.title(tit)
	return ax,ax_fst,ax_last



#Center spike from max
def center(events,ms,dt=0.1):
	mx_index = np.argmax(events) #index of maximum V value (spike)
	ms_points = ms /dt #Number of points corresponding to the iteration
	# ms_points = ms
	

	# print(mx_index)
	ini = int(mx_index-ms_points) #init as max point - number of points. 
	end = int(mx_index+ms_points) #end as max point + number of points. 

	###Beta func: Beware in models
	if mx_index!=0: #ignore artefacts
		#Adjust window when there are not enough points 
		if(ini < 0):
			app = np.full(abs(ini),events[0]) 
			events =np.insert(events,0,app) #Add events at the begining
			return center(events,ms,dt) #re-center
		if(end > events.shape[0]):
			app = np.full(end-events.shape[0],events[-1]) #Add events at the end
			events = np.insert(events,events.shape[0],app) #re-center
			return center(events,ms,dt)

	####

	return events[ini:end]



def no_drift(events):
	if(events.shape[0]!=0):
		mn = np.min(events)
		if mn != 0:
			events = events-mn
	
	return events