import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
from superpos_functions import *

plt.rcParams.update({'font.size': 15})


if len(sys.argv) ==3:
	path = sys.argv[1]
	path_control_pre = path+"_control_pre_events.txt"
	path_laser = path+"_laser_events.txt"
	path_control_pos = path+"_control_pos_events.txt"
	width = int(sys.argv[2])

else:
	print("Use: python3 superpos_from_events_3.py path width ")
	exit()



os.system("sed -i 's/\,/./g' "+path_control_pre) #changing , to . to read floats not strings
os.system("sed -i 's/\,/./g' "+path_laser) #changing , to . to read floats not strings
os.system("sed -i 's/\,/./g' "+path_control_pos) #changing , to . to read floats not strings

#Each row contains voltage values of the corresponding event.
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

#------------------------------------------------

#Labels for Control-Control

# label1 = "First control. N. spikes: %d"%(n_control)
# label2 = "Last control. N. spikes: %d"%(n_laser)
# path = path[:path.find("exp")] +"first_last_control"
#------------------------------------------------

#Labels for Control-Laser
# label1 = "First laser. N. spikes: %d"%(n_control)
# label2 = "Last laser. N. spikes: %d"%(n_laser)
# path = path[:path.find("exp")] +"first_last_laser"
#------------------------------------------------

# #Individual plots
# plot_events(control_pre_events,col='b',tit=label1,ms=width)
# plt.show()
# plot_events(laser_events,col='r',tit=label2,ms=width)
# plt.show()
# plot_events(control_pos_events,col='g',tit=label3,ms=width)
# plt.show()

# #ControlPre-Laser

# ax1= plot_events(control_pre_events,'b',path,ms=width)
# ax2=plot_events(laser_events,'r',path,ms=width)

# plt.legend([ax1,ax2],[label1,label2])
# plt.savefig(path +"pre_laser.png")
# plt.show()

# #ControlPos-Laser
# ax1= plot_events(control_pos_events,'g',path,ms=width)
# ax2=plot_events(laser_events,'r',path,ms=width)

# plt.legend([ax1,ax2],[label1,label2])
# plt.savefig(path +"pos_laser.png")
# plt.show()


# #ControlPre-ControlPos
# ax1= plot_events(control_pre_events,'b',path,ms=width)
# ax3=plot_events(control_pos_events,'g',path,ms=width)

# plt.legend([ax1,ax3],[label1,label3])
# plt.savefig(path +"_controls.png")
# plt.show()

# #Pre-Laser-Pos

# ax1= plot_events(control_pre_events,'b',path,ms=width)
# ax2=plot_events(laser_events,'r',path,ms=width)
# ax3= plot_events(control_pos_events,'g',path,ms=width)
# plt.legend([ax1,ax2,ax3],[label1,label2,label3])
# plt.savefig(path +"pre_laser_pos.png")
# plt.show()

#########################################################
######## Plot in grid ################################
########################################################
plt.figure(figsize=(20,15))
plt.tight_layout()
rows = 3
columns= 3
#Individual plots
plt.subplot(rows,columns,1)

ax1,ax_fst,ax_last =plot_events(control_pre_events,col='b',tit=label1,ms=width)
plt.legend([ax_fst,ax_last],["First spike","Last spike"])

plt.subplot(rows,columns,2)
ax1,ax_fst,ax_last =plot_events(laser_events,col='r',tit=label2,ms=width)
plt.legend([ax_fst,ax_last],["First spike","Last spike"])

plt.subplot(rows,columns,3)
ax1,ax_fst,ax_last =plot_events(control_pos_events,col='g',tit=label3,ms=width)
plt.legend([ax_fst,ax_last],["First spike","Last spike"])

#ControlPre-Laser
plt.tight_layout()

plt.subplot(rows,columns,4)
ax1,ax_fst,ax_last= plot_events(control_pre_events,'b',tit="ControlPre-Laser",ms=width)
ax2,ax_fst,ax_last=plot_events(laser_events,'r',tit="ControlPre-Laser",ms=width)

plt.legend([ax1,ax2],[label1,label2],loc="lower left")
# plt.savefig(path +"pre_laser.png")
# plt.show()

plt.tight_layout()
#ControlPos-Laser

plt.subplot(rows,columns,5)
ax1,ax_fst,ax_last= plot_events(control_pos_events,'g',tit="ControlPos-Laser",ms=width)
ax2,ax_fst,ax_last=plot_events(laser_events,'r',tit="ControlPos-Laser",ms=width)

plt.legend([ax1,ax2],[label1,label2],loc="lower left")
# plt.savefig(path +"pos_laser.png")
# plt.show()

plt.tight_layout()

#ControlPre-ControlPos

plt.tight_layout()
plt.subplot(rows,columns,6)
ax1,ax_fst,ax_last= plot_events(control_pre_events,'b',tit="ControlPre-ControlPos",ms=width)
ax3,ax_fst,ax_last=plot_events(control_pos_events,'g',tit="ControlPre-ControlPos",ms=width)

plt.legend([ax1,ax3],[label1,label3],loc="lower left")
# plt.savefig(path +"_controls.png")
# plt.show()

#Pre-Laser-Pos

plt.tight_layout()
plt.subplot(rows,columns,8)
ax1,ax_fst,ax_last= plot_events(control_pre_events,'b',tit="Pre-Laser-Pos",ms=width)
ax2,ax_fst,ax_last=plot_events(laser_events,'r',tit="Pre-Laser-Pos",ms=width)
ax3,ax_fst,ax_last= plot_events(control_pos_events,'g',tit="Pre-Laser-Pos",ms=width)
plt.legend([ax1,ax2,ax3],[label1,label2,label3],loc="lower left")


plt.suptitle(path)
plt.tight_layout()
plt.savefig(path +".png")
plt.show()
