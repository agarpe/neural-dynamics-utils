
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
		if(row_i==0):
			ax_fst,=plt.plot(row,color=fst_color,linewidth=1.5)
		elif(row_i==events.shape[0]-1):
			ax_last,=plt.plot(row,color=last_color,linewidth=1.5)
		else:
			ax,=plt.plot(row,color=col,linewidth=0.1)	
		# ax,=plt.plot(row,color=col,linewidth=0.1)
			ax,=plt.plot(row,linewidth=0.1)
	plt.title(tit)
	return ax,ax_fst,ax_last



#Center spike from max
def center(events,ms,dt=0.1):
	mx_index = np.argmax(events) #index of maximum V value (spike)
	ms_points = ms /dt #Number of points corresponding to the iteration
	
	ini = int(mx_index-ms_points) #init as max point - number of points. 
	end = int(mx_index+ms_points) #end as max point + number of points. 

	return events[ini:end]



def no_drift(events):
	if(events.shape[0]!=0):
		mn = np.min(events)
		if mn != 0:
			events = events-mn
	
	return events