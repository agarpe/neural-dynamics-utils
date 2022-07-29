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
ap.add_argument("-c1", "--color1", required=False, default='b', help="Color for first file")
ap.add_argument("-c2", "--color2", required=False, default='r', help="Color for second file")
ap.add_argument("-ti", "--title", required=True, help="Title of the resulting plot")
ap.add_argument("-mean","--mean",required=False, default='n',help="When == 'y'. Plot mean of all spikes and not spike per spike.")
ap.add_argument("-ali","--align",required=False, default='peak',help="Choose alignment mode. 'peak', 'min', 'max', 'ini', 'end'")
ap.add_argument("-sa", "--save", required=False, default='y', help="Option to save plot file")
ap.add_argument("-sh", "--show", required=False, default='y', help="Option to show plot file")
ap.add_argument("-st", "--stats", required=False, default='y', help="Option to save stats pkl file")
ap.add_argument("-nr", "--rows", required=False, default=2, 
				help="Number of rows. 2 for single and comparations. 3 for prevs and single subplot with the three of them")
ap.add_argument("-sc", "--scale", required=False, default=1, help="Scale for Volts. Signal*scale")

args = vars(ap.parse_args())




path_control = args["path1"]
path_laser = args["path2"]
width = int(args["window_width"])
label1 = args["label1"]
label2 = args["label2"]
color1 = args["color1"]
color2 = args["color2"]
title = args["title"]

mode = args['align']
rows = int(args['rows'])
scale = float(args['scale'])

show = True if args['show']=='y' else False 
save = True if args['save']=='y' else False 
stats = True if args['save']=='y' else False 


plot_func = plot_events_mean if args['mean']=='y' else plot_events 

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
control_events=control_events.values*scale
laser_events=laser_events.values*scale

print(control_events.shape)
print(laser_events.shape)


# #Labels for Control-Laser
label1 = label1+" "+str(n_control)
label2 = label2+" "+str(n_laser)

log_1={}
log_2={}


#------------------------------------------------
# Plot 

#Individual plots
if plot_func==plot_events:
	legends = ["First spike","Last spike"]
else:
	legends = []

if rows >1:

	plt.figure(figsize=(20,15))
	plt.tight_layout()

	plt.subplot(2,2,1)

	ax1,ax_fst,ax_last=plot_func(control_events,col=color1,tit=label1,width_ms=width,df_log=log_1, mode=mode)
	plt.legend([ax_fst,ax_last],legends)
	plt.xlabel("Time (ms)")
	plt.ylabel("Voltage (mV)")


	plt.subplot(2,2,2)
	ax1,ax_fst,ax_last=plot_func(laser_events,col=color2,tit=label2,width_ms=width,df_log=log_2, mode=mode)
	plt.legend([ax_fst,ax_last],legends)
	plt.xlabel("Time (ms)")
	plt.ylabel("Voltage (mV)")


if rows ==1:
	plt.figure(figsize=(10,rows*10))
else:
	plt.subplot(2,2,3)
#ControlPre-Laser
plt.tight_layout()

ax1,ax_fst,ax_last= plot_func(control_events,color1,tit=label1+"-"+label2,width_ms=width, df_log=log_1, mode=mode)
ax2,ax_fst,ax_last=plot_func(laser_events,color2,tit=label1+"-"+label2,width_ms=width, df_log=log_2, mode=mode)

plt.legend([ax1,ax2,ax_fst,ax_last],[label1,label2,legends])
plt.tight_layout()
plt.xlabel("Time (ms)")
plt.ylabel("Voltage (mV)")





path = path_control

# path = path[:path.find("exp")] +title

if args['mean'] == 'y':
	m = 'mean'
else:
	m = ''
path = path[:-4]+"_"+title + "_" + mode +"_" + m + '_' + str(rows)

plt.suptitle(title)
plt.tight_layout(rect=[0, 0, 1, 0.95])

if save:
	plt.savefig(path +".png")
	plt.savefig(path +".eps",format='eps', dpi=1200)
if show:
	plt.show()


if stats:
	#Saving dataframes
	print("saving dataframes")

	df = create_dataframe([log_1,log_2],[label1,label2])
	print(df.describe())
	df.to_pickle(path+"_info.pkl")
