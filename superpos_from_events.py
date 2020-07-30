import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
from superpos_functions import *

plt.rcParams.update({'font.size': 17})


if len(sys.argv) ==4:
	path_control = sys.argv[1]
	path_laser = sys.argv[2]
	path = path_control[:path_control.find("control")] 
	width = int(sys.argv[3])
else:
	print("Use1: python3 superpos_from_events.py events_1_path.txt events_2_path.txt")
	exit()



os.system("sed -i 's/\,/./g' "+path_control) #changing , to . to read floats not strings
os.system("sed -i 's/\,/./g' "+path_laser) #changing , to . to read floats not strings

#Each row contains voltage values of the corresponding event.
control_events =  pd.read_csv(path_control, delimiter = "\t",skiprows=0,header=None)
laser_events =  pd.read_csv(path_laser, delimiter = "\t",skiprows=0,header=None)

n_control = len(control_events.index)
n_laser = len(laser_events.index)

#Remove last column NaN values
control_events=control_events.iloc[:, :-1] 
laser_events=laser_events.iloc[:, :-1]

#Parse to array
control_events=control_events.values
laser_events=laser_events.values


# #Labels for Control-Laser
# label1 = "Control. N. spikes: %d"%(n_control)
# label2 = "Laser. N. spikes: %d"%(n_laser)
#------------------------------------------------

#Labels for Control-Control

label1 = "First control. N. spikes: %d"%(n_control)
label2 = "Last control. N. spikes: %d"%(n_laser)
path = path[:path.find("exp")] +"first_second_control"
color2 = 'g'
#------------------------------------------------

# Labels for Laser-Laser
# label1 = "First laser. N. spikes: %d"%(n_control)
# label2 = "Last laser. N. spikes: %d"%(n_laser)
# path = path[:path.find("exp")] +"first_last_laser"
# color2 = 'r'
#------------------------------------------------


# plot_events(control_events,col='b',tit=label1,ms=width)
# plt.show()
# plot_events(laser_events,col='r',tit=label2,ms=width)
# plt.show()

# ax1= plot_events(control_events,'b',path,ms=width)
# ax2=plot_events(laser_events,'r',path,ms=width)

# plt.legend([ax1,ax2],[label1,label2])
# plt.savefig(path +".png")
# plt.show()


plt.figure(figsize=(20,15))
plt.tight_layout()
#Individual plots
plt.subplot(2,2,1)

ax1,ax_fst,ax_last=plot_events(control_events,col='b',tit=label1,ms=width)
plt.legend([ax_fst,ax_last],["First spike","Last spike"])


plt.subplot(2,2,2)
ax1,ax_fst,ax_last=plot_events(laser_events,col=color2,tit=label2,ms=width)
plt.legend([ax_fst,ax_last],["First spike","Last spike"])


#ControlPre-Laser
plt.tight_layout()

plt.subplot(2,2,3)
ax1,ax_fst,ax_last= plot_events(control_events,'b',tit="ControlPre-Laser",ms=width)
ax2,ax_fst,ax_last=plot_events(laser_events,color2,tit="ControlPre-Laser",ms=width)

plt.legend([ax1,ax2,ax_fst,ax_last],[label1,label2,"First spike","Last spike"])
plt.tight_layout()

# plt.subplot(2,2,4)
# ax1,ax_fst,ax_last= plot_events(control_events,'b',tit="ControlPre-Laser",ms=width)
# ax2,ax_fst,ax_last=plot_events(laser_events,color2,tit="ControlPre-Laser",ms=width)

plt.suptitle(path)
plt.tight_layout()
plt.savefig(path +".png")
plt.show()
