import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys


# def plot_events(events,col,tit="Spikes"):
# 	# events=events.dropna()
# 	ax=0
# 	for row in range(len(events.T.index)):
# 		try:
# 			ax,=plt.plot(events[:][row],color=col)
# 		except:
# 			pass
# 	plt.title(tit)
# 	return ax


def plot_events(events,col,tit="Spikes"):
	ax=0
	for row in range(events.shape[0]):
		try:
			ax,=plt.plot(events[row][:],color=col,linewidth=0.1)
		except:
			pass
	plt.title(tit)
	return ax


def center(events,ms):
	mx_index = np.argmax(events)
	ms_points = ms /0.1 #Number of points corresponding to the iteration
	
	return events[ms_points:ms_points]




if len(sys.argv) !=3:
	print("Use: python3 superpos_from_events.py control_events_path laser_events_path")
	exit()

path_control = sys.argv[1]
path_laser = sys.argv[2]
time_scale=1


os.system("sed -i 's/\,/./g' "+path_control) #changing , to . to read floats not strings
os.system("sed -i 's/\,/./g' "+path_laser) #changing , to . to read floats not strings

#Each row contains voltage values of the corresponding event.
control_events =  pd.read_csv(path_control, delimiter = "\t",skiprows=0,header=None)
laser_events =  pd.read_csv(path_laser, delimiter = "\t",skiprows=0,header=None)

n_control = len(control_events.index)
n_laser = len(laser_events.index)

control_events=control_events.values
laser_events=laser_events.values



plot_events(control_events,col='b',tit="Control spikes %d"%(n_control))
plt.show()
plot_events(laser_events,col='r',tit="Laser spikes %d"%(n_laser))
plt.show()

ax1= plot_events(control_events,'b')
ax2=plot_events(laser_events,'r')

label1 = "Control spikes %d"%(n_control)
label2 = "Laser spikes %d"%(n_laser)
plt.legend([ax1,ax2],[label1,label2])
plt.show()
