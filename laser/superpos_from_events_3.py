#Superpos plots from different spikes events using functions in superpos_functions.py
#Script prepared for 3 different trials (e.g. control pre, laser, control pos)
#Generates a plot of suplots and save the spike amplitude dataset as path_info.plk
#Example of use:
#	python3 superpos_from_events_3.py pruebas/exp4_5400_50f 50

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
from superpos_functions import *
import itertools

plt.rcParams.update({'font.size': 19})

# import argparse

# ap = argparse.ArgumentParser()
# ap.add_argument("-p", "--path", required=True, help="Path to the file to plot")
# ap.add_argument("-fcp", "--file_events", required=True, help="Events file name")
# ap.add_argument("-sa", "--save", required=False, default='0', help="Save to file and the name")
# ap.add_argument("-st", "--start", required=False, default=0, help="Start second")
# ap.add_argument("-en", "--end", required=False, default='None', help="End second")
# ap.add_argument("-fr", "--frecuency", required=False, default=10000, help="Sampling freq of -fs")
# ap.add_argument("-mo", "--mode", required=False, default=2, help="Mode '1' points appear with the signal. Mode '2' points always on display")
# ap.add_argument("-cu", "--current", required=False, default=0, help="'1' Display current injected. '0' ignores current")
# ap.add_argument("-fps", "--fps", required=False, default=24, help="Frames per second")
# ap.add_argument("-dpi", "--dpi", required=False, default=100, help="Dots per inch")
# args = vars(ap.parse_args())
# save = args['save']



if len(sys.argv) ==3:
	path = sys.argv[1]
	path_control_pre = path+"_control_pre_waveform.txt"
	path_laser = path+"_laser_waveform.txt"
	path_control_pos = path+"_control_pos_waveform.txt"
	width = int(sys.argv[2])
	show = False
	save = True
else:
	print("Use: python3 superpos_from_events_3.py path width ")
	exit()


os.system("sed -i 's/\,/./g' "+path_control_pre) #changing , to . to read floats not strings
os.system("sed -i 's/\,/./g' "+path_laser) #changing , to . to read floats not strings
os.system("sed -i 's/\,/./g' "+path_control_pos) #changing , to . to read floats not strings

try:
	#Each row contains Voltage values of the corresponding event.
	control_pre_events = read_from_events(path_control_pre,max_cols=300,dt=0.1)
	laser_events =  read_from_events(path_laser,max_cols=300,dt=0.1)
	control_pos_events =  read_from_events(path_control_pos,max_cols=300,dt=0.1)
except:
	print("Error: file not found")
	exit()


n_control_pre = len(control_pre_events.index)
n_laser = len(laser_events.index)
n_control_pos = len(control_pos_events.index)

#Remove last column NaN values
control_pre_events=control_pre_events.iloc[:-1, :-1] 
laser_events=laser_events.iloc[:-1, :-1]
control_pos_events=control_pos_events.iloc[:-1, :-1] 

#Parse to array
control_pre_events=control_pre_events.values[:-1]
laser_events=laser_events.values
control_pos_events=control_pos_events.values


#Labels for Control-Laser
label1 = "Control pre. N. spikes: %d"%(n_control_pre)
label2 = "Laser. N. spikes: %d"%(n_laser)
label3 = "Control pos. N. spikes: %d"%(n_control_pos)

#Dafaframes and logs
control_pre_log = []
laser_log = []
control_pos_log = []


#------------------------------------------------

#########################################################
######## Plot in grid ################################
########################################################
plt.figure(figsize=(30,25))
rows = 3
columns= 3


#Individual plots
plt.subplot(rows,columns,1)
ax1,ax_fst,ax_last =plot_events(control_pre_events,col='b',tit=label1,width_ms=width,amplitude_log=control_pre_log,show_amplitudes=False)
set_plot_info([ax_fst,ax_last],["First spike","Last spike"])

plt.subplot(rows,columns,2)
ax1,ax_fst,ax_last =plot_events(laser_events,col='r',tit=label2,width_ms=width,amplitude_log=laser_log,show_amplitudes=False)
set_plot_info([ax_fst,ax_last],["First spike","Last spike"])

plt.subplot(rows,columns,3)
ax1,ax_fst,ax_last =plot_events(control_pos_events,col='g',tit=label3,width_ms=width,amplitude_log=control_pos_log,show_amplitudes=False)
set_plot_info([ax_fst,ax_last],["First spike","Last spike"])


#ControlPre-Laser

plt.subplot(rows,columns,4)
ax1,ax_fst,ax_last= plot_events(control_pre_events,'b',tit="ControlPre-Laser",width_ms=width)
ax2,ax_fst,ax_last=plot_events(laser_events,'r',tit="ControlPre-Laser",width_ms=width)

set_plot_info([ax1,ax2],[label1,label2],loc="lower left")


#ControlPos-Laser

plt.subplot(rows,columns,5)
ax1,ax_fst,ax_last= plot_events(control_pos_events,'g',tit="ControlPos-Laser",width_ms=width)
ax2,ax_fst,ax_last=plot_events(laser_events,'r',tit="ControlPos-Laser",width_ms=width)

set_plot_info([ax1,ax2],[label3,label2],loc="lower left")

#ControlPre-ControlPos

plt.subplot(rows,columns,6)
ax1,ax_fst,ax_last= plot_events(control_pre_events,'b',tit="ControlPre-ControlPos",width_ms=width)
ax3,ax_fst,ax_last=plot_events(control_pos_events,'g',tit="ControlPre-ControlPos",width_ms=width)

set_plot_info([ax1,ax3],[label1,label3],loc="lower left")

#Pre-Laser-Pos

plt.subplot(rows,columns,8)
ax1,ax_fst,ax_last= plot_events(control_pre_events,'b',tit="Pre-Laser-Pos",width_ms=width)
ax2,ax_fst,ax_last=plot_events(laser_events,'r',tit="Pre-Laser-Pos",width_ms=width)
ax3,ax_fst,ax_last= plot_events(control_pos_events,'g',tit="Pre-Laser-Pos",width_ms=width)

set_plot_info([ax1,ax2,ax3],[label1,label2,label3],loc="lower left")


plt.suptitle(path) #general title
plt.tight_layout(rect=[0, 0, 1, 0.95]) #tight with upper title
if(save):
	plt.savefig(path +".png")
if(show):
	plt.show()



# Saving logs into dataframe


#zip amplitude logs saving None values
data_tuples=list(itertools.zip_longest(control_pre_log,laser_log,control_pos_log))
# print(data_tuples)
df = pd.DataFrame(data_tuples, columns=['control_pre','laser','control_pos'])

print(df.describe())
print("Mean difference form control to control:",df['control_pre'].mean()-df['control_pos'].mean())
print("Mean difference form control pre to laser:",df['control_pre'].mean()-df['laser'].mean())
print("Mean difference form control pos to laser:",df['control_pos'].mean()-df['laser'].mean())

#Saving amplitude dataframes
df.to_pickle(path+"_info.pkl")