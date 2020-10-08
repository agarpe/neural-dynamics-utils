import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
from superpos_functions import *
import itertools
# plt.rcParams['figure.constrained_layout.use'] = True

plt.rcParams.update({'font.size': 19})

if len(sys.argv) ==3:
	path = sys.argv[1]
	path_control_pre = path+"_control_pre_events.txt"
	path_laser = path+"_laser_events.txt"
	path_control_pos = path+"_control_pos_events.txt"
	width = int(sys.argv[2])
	show = True

else:
	print("Use: python3 superpos_from_events_3.py path width ")
	exit()


os.system("sed -i 's/\,/./g' "+path_control_pre) #changing , to . to read floats not strings
os.system("sed -i 's/\,/./g' "+path_laser) #changing , to . to read floats not strings
os.system("sed -i 's/\,/./g' "+path_control_pos) #changing , to . to read floats not strings

#Each row contains Voltage values of the corresponding event.
control_pre_events =  pd.read_csv(path_control_pre, delimiter = "\t",skiprows=0,header=None)
laser_events =  pd.read_csv(path_laser, delimiter = "\t",skiprows=0,header=None)
control_pos_events =  pd.read_csv(path_control_pos, delimiter = "\t",skiprows=0,header=None)

n_control_pre = len(control_pre_events.index)
n_laser = len(laser_events.index)
n_control_pos = len(control_pos_events.index)

#Remove last column NaN values
control_pre_events=control_pre_events.iloc[:, :-1] 
laser_events=laser_events.iloc[:, :-1]
control_pos_events=control_pos_events.iloc[:, :-1] 

#Parse to array
control_pre_events=control_pre_events.values
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
plt.legend([ax_fst,ax_last],["First spike","Last spike"])
plt.xlabel("Time (ms)")
plt.ylabel("Voltage (mV)")

plt.subplot(rows,columns,2)
ax1,ax_fst,ax_last =plot_events(laser_events,col='r',tit=label2,width_ms=width,amplitude_log=laser_log)
plt.legend([ax_fst,ax_last],["First spike","Last spike"])
plt.xlabel("Time (ms)")
plt.ylabel("Voltage (mV)")

plt.subplot(rows,columns,3)
ax1,ax_fst,ax_last =plot_events(control_pos_events,col='g',tit=label3,width_ms=width,amplitude_log=control_pos_log)
plt.legend([ax_fst,ax_last],["First spike","Last spike"])
plt.xlabel("Time (ms)")
plt.ylabel("Voltage (mV)")


#ControlPre-Laser

plt.subplot(rows,columns,4)
ax1,ax_fst,ax_last= plot_events(control_pre_events,'b',tit="ControlPre-Laser",width_ms=width)
ax2,ax_fst,ax_last=plot_events(laser_events,'r',tit="ControlPre-Laser",width_ms=width)

plt.legend([ax1,ax2],[label1,label2],loc="lower left")
plt.xlabel("Time (ms)")
plt.ylabel("Voltage (mV)")


#ControlPos-Laser

plt.subplot(rows,columns,5)
ax1,ax_fst,ax_last= plot_events(control_pos_events,'g',tit="ControlPos-Laser",width_ms=width)
ax2,ax_fst,ax_last=plot_events(laser_events,'r',tit="ControlPos-Laser",width_ms=width)

plt.legend([ax1,ax2],[label3,label2],loc="lower left")
plt.xlabel("Time (ms)")
plt.ylabel("Voltage (mV)")

#ControlPre-ControlPos

plt.subplot(rows,columns,6)
ax1,ax_fst,ax_last= plot_events(control_pre_events,'b',tit="ControlPre-ControlPos",width_ms=width)
ax3,ax_fst,ax_last=plot_events(control_pos_events,'g',tit="ControlPre-ControlPos",width_ms=width)

plt.legend([ax1,ax3],[label1,label3],loc="lower left")
plt.xlabel("Time (ms)")
plt.ylabel("Voltage (mV)")

#Pre-Laser-Pos

plt.subplot(rows,columns,8)
ax1,ax_fst,ax_last= plot_events(control_pre_events,'b',tit="Pre-Laser-Pos",width_ms=width)
ax2,ax_fst,ax_last=plot_events(laser_events,'r',tit="Pre-Laser-Pos",width_ms=width)
ax3,ax_fst,ax_last= plot_events(control_pos_events,'g',tit="Pre-Laser-Pos",width_ms=width)
plt.legend([ax1,ax2,ax3],[label1,label2,label3],loc="lower left")
plt.xlabel("Time (ms)")
plt.ylabel("Voltage (mV)")



plt.suptitle(path) #general title
plt.tight_layout(rect=[0, 0, 1, 0.95]) #tight with upper title
plt.savefig(path +".png")
if(show):
	plt.show()



# Saving logs into dataframe


#zip amplitude logs saving None values
data_tuples=list(itertools.zip_longest(control_pre_log,laser_log,control_pos_log))
# print(data_tuples)
df = pd.DataFrame(data_tuples, columns=['control_pre','laser','control_pos'])

print(df.describe())

#Saving amplitude dataframes

df.to_pickle(path+"_info.pkl")