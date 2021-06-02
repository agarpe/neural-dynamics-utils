import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
from superpos_functions import *
import argparse

plt.rcParams.update({'font.size': 17})

ap = argparse.ArgumentParser()
ap.add_argument("-p1", "--path1", required=True, help="First path to wafeforms file")
ap.add_argument("-p2", "--path2", required=True, help="Second path to wafeforms file")
ap.add_argument("-w", "--window_width", required=True, help="Half window width in ms")
ap.add_argument("-l1", "--label1", required=True, help="Label of first file")
ap.add_argument("-l2", "--label2", required=True, help="Label of second file")
ap.add_argument("-c2", "--color2", required=True, help="Color for second file")
ap.add_argument("-ti", "--title", required=True, help="Title of the resulting plot")
ap.add_argument("-sa", "--save", required=False, default='y', help="Option to save plot file")
ap.add_argument("-sh", "--show", required=False, default='y', help="Option to show plot file")
ap.add_argument("-st", "--stats", required=False, default='y', help="Option to save stats pkl file")
args = vars(ap.parse_args())



path_control = args["path1"]
path_laser = args["path2"]
width = int(args["window_width"])
label1 = args["label1"]
label2 = args["label2"]
color2 = args["color2"]
title = args["title"]
show = True if args['show']=='y' else False 
save = True if args['save']=='y' else False 
stats = True if args['save']=='y' else False 



# #Each row contains voltage values of the corresponding event.
try:
	control_events = read_from_events(path_control,max_cols=300,dt=0.1,dataview=True)
except:
	print("Error: file 1 not found")
	exit()
try:
	laser_events = read_from_events(path_laser,max_cols=300,dt=0.1,dataview=True)
except:
	print("Error: file 2 not found")
	exit()

n_control = len(control_events.index)
n_laser = len(laser_events.index)


#Parse to array
control_events=control_events.values
laser_events=laser_events.values

print(control_events.shape)
print(laser_events.shape)


# #Labels for Control-Laser
label1 = label1+" "+str(n_control)
label2 = label2+" "+str(n_laser)

log_1={}
log_2={}


#------------------------------------------------
# Plot 

plt.figure(figsize=(20,15))
plt.tight_layout()

#Individual plots
plt.subplot(2,2,1)

ax1,ax_fst,ax_last=plot_events(control_events,col='b',tit=label1,width_ms=width,df_log=log_1)
plt.legend([ax_fst,ax_last],["First spike","Last spike"])
plt.xlabel("Time (ms)")
plt.ylabel("Voltage (mV)")


plt.subplot(2,2,2)
ax1,ax_fst,ax_last=plot_events(laser_events,col=color2,tit=label2,width_ms=width,df_log=log_2)
plt.legend([ax_fst,ax_last],["First spike","Last spike"])
plt.xlabel("Time (ms)")
plt.ylabel("Voltage (mV)")


#ControlPre-Laser
plt.tight_layout()

plt.subplot(2,2,3)
ax1,ax_fst,ax_last= plot_events(control_events,'b',tit="ControlPre-Laser",width_ms=width)
ax2,ax_fst,ax_last=plot_events(laser_events,color2,tit="ControlPre-Laser",width_ms=width)

plt.legend([ax1,ax2,ax_fst,ax_last],[label1,label2,"First spike","Last spike"])
plt.tight_layout()
plt.xlabel("Time (ms)")
plt.ylabel("Voltage (mV)")

path = path_control

# path = path[:path.find("exp")] +title

path = path[:-4]+"_"+title

plt.suptitle(title)
plt.tight_layout(rect=[0, 0, 1, 0.95])

if save:
	plt.savefig(path +".png")
if show:
	plt.show()


if stats:
	#Saving dataframes
	print("saving dataframes")

	df = create_dataframe([log_1,log_2],[label1,label2])
	print(df.describe())
	df.to_pickle(path+"_info.pkl")
