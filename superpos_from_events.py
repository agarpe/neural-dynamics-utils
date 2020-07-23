import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys

plt.rcParams.update({'font.size': 17})

def plot_events(events,col,tit,ms=50):
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

	for row_i in range(events.shape[0]):
		row = center(events[row_i,:],ms) #center spike from max
		row = no_drift(row) #adjust drift
		if(row_i==0):
			ax_fst,=plt.plot(row,color=fst_color,linewidth=2)
		elif(row_i==events.shape[0]-1):
			ax_last,=plt.plot(row,color=last_color,linewidth=2)
		else:
			ax,=plt.plot(row,color=col,linewidth=0.1)	
		# ax,=plt.plot(row,color=col,linewidth=0.1)
			ax,=plt.plot(row,linewidth=0.1)
	plt.title(tit)
	return ax,ax_fst,ax_last


#Center spike from max
def center(events,ms):
	mx_index = np.argmax(events) #index of maximum V value (spike)
	ms_points = ms /0.1 #Number of points corresponding to the iteration
	
	ini = int(mx_index-ms_points) #init as max point - number of points. 
	end = int(mx_index+ms_points) #end as max point + number of points. 

	return events[ini:end]

def no_drift(events):
	if(events.shape[0]!=0):
		mn = np.min(events)
		if mn != 0:
			events = events-mn
	
	return events



if len(sys.argv) ==3:
	path = sys.argv[1]
	path_control = path+"_control_pre_events.txt"
	path_laser = path+"_laser_events.txt"
	width = 10

elif len(sys.argv) ==4:
	path_control = sys.argv[1]
	path_laser = sys.argv[2]
	path = path_control[:path_control.find("control")] 
	width = 10
else:
	print("Use1: python3 superpos_from_events.py control_events_path.txt laser_events_path.txt")
	print("Use2: python3 superpos_from_events.py path ")
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


#Labels for Control-Laser
label1 = "Control. N. spikes: %d"%(n_control)
label2 = "Laser. N. spikes: %d"%(n_laser)
#------------------------------------------------

#Labels for Control-Laser

# label1 = "First control. N. spikes: %d"%(n_control)
# label2 = "Last control. N. spikes: %d"%(n_laser)
# path = path[:path.find("exp")] +"first_last_control"
#------------------------------------------------

#Labels for Control-Laser
# label1 = "First laser. N. spikes: %d"%(n_control)
# label2 = "Last laser. N. spikes: %d"%(n_laser)
# path = path[:path.find("exp")] +"first_last_laser"
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
ax1,ax_fst,ax_last=plot_events(laser_events,col='r',tit=label2,ms=width)
plt.legend([ax_fst,ax_last],["First spike","Last spike"])


#ControlPre-Laser
plt.tight_layout()

plt.subplot(2,2,3)
ax1,ax_fst,ax_last= plot_events(control_events,'b',tit="ControlPre-Laser",ms=width)
ax2,ax_fst,ax_last=plot_events(laser_events,'r',tit="ControlPre-Laser",ms=width)

plt.legend([ax1,ax2,ax_fst,ax_last],[label1,label2,"First spike","Last spike"])
plt.tight_layout()

plt.subplot(2,2,4)
ax1,ax_fst,ax_last= plot_events(control_events,'b',tit="ControlPre-Laser",ms=width)
ax2,ax_fst,ax_last=plot_events(laser_events,'r',tit="ControlPre-Laser",ms=width)

plt.suptitle(path)
plt.tight_layout()
plt.savefig(path +".png")
plt.show()
