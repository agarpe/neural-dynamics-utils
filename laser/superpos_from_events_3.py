#Superpos plots from different spikes events using functions in superpos_functions.py
#Script prepared for 3 different trials (e.g. control pre, laser, control pos)
#Generates a plot of suplots and save the spike duration dataset as path_info.plk
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


# def create_dataframe(dicts,prefixes):
# 	if len(dicts) != len(prefixes):
# 		print("Error creating dataframe, dicts and prefixes with diffrent shapes")
# 		return {}

# 	dfs = []

# 	for d,p in zip(dicts,prefixes):
# 		df = pd.DataFrame.from_dict(d, orient='index')
# 		df = df.transpose()
# 		dfs.append(df.add_prefix(p))

# 	df = pd.concat(dfs,axis=1)

# 	return df

# import argparse

# ap = argparse.ArgumentParser()
# ap.add_argument("-p", "--path", required=True, help="Path to the experiment trial")
# ap.add_argument("-w", "--window_width", required=True, help="Half window width in ms")
# ap.add_argument("-sa", "--save", required=False, default='y', help="Option to save plot file")
# ap.add_argument("-sh", "--show", required=False, default='y', help="Option to show plot file")
# ap.add_argument("-st", "--stats", required=False, default='y', help="Option to save stats pkl file")
# args = vars(ap.parse_args())


# path = args['path']
# path_control_pre = path+"_control_pre_waveform.txt"
# path_laser = path+"_laser_waveform.txt"
# path_control_pos = path+"_control_pos_waveform.txt"
# width = int(args['window_width'])
# show = args['show']
# save = args['save']

if len(sys.argv) ==3:
	path = sys.argv[1]
	path_control_pre = path+"_control_pre_waveform.txt"
	path_laser = path+"_laser_waveform.txt"
	path_control_pos = path+"_control_pos_waveform.txt"
	width = int(sys.argv[2])
	show = 'n'
	save = 'y'
	stats = 'y'
else:
	print("Use: python3 superpos_from_events_3.py path width ")
	exit()


print("\nSuperposing from ",path)

print("Reading events files...")

try:
	#Each row contains Voltage values of the corresponding event.
	control_pre_events = read_from_events(path_control_pre,max_cols=300,dt=0.1)
	laser_events =  read_from_events(path_laser,max_cols=300,dt=0.1)
	control_pos_events =  read_from_events(path_control_pos,max_cols=300,dt=0.1)
except:
	print("Error: file not found")
	print(path_control_pre)
	exit()


n_control_pre = len(control_pre_events.index)
n_laser = len(laser_events.index)
n_control_pos = len(control_pos_events.index)


#Parse to array
control_pre_events=control_pre_events.values
laser_events=laser_events.values
control_pos_events=control_pos_events.values


#Labels for Control-Laser
label1 = "Control pre. N. spikes: %d"%(n_control_pre)
label2 = "Laser. N. spikes: %d"%(n_laser)
label3 = "Control pos. N. spikes: %d"%(n_control_pos)

#Dafaframes and logs
control_pre_log ={}
laser_log ={}
control_pos_log ={}

# color_pre = 'b'
# color_laser = 'r'
# color_pos = 'g'

#Error: invalid color???
color_pre = (Color("lightcyan"),Color("cornflowerblue"))
color_pos = (Color("skyblue"),Color("darkblue"))
color_laser = (Color("lightsalmon"),Color("darkred"))



# blue = Color("blue")
# color = blue
# color.luminance = luminances[i%(len(files))]
# color = color.hex_l
colors = {'b':['cyan','darkblue'],'r':['coral','maroon'],'g':['lime','darkgreen']}

#------------------------------------------------

#########################################################
######## Plot in grid ################################
########################################################
plt.figure(figsize=(30,25))
rows = 3
columns= 3



#Individual plots
plt.subplot(rows,columns,1)
ax1,ax_fst,ax_last =plot_events(control_pre_events,col=color_pre,tit=label1,width_ms=width,df_log=control_pre_log,show_durations=False)
set_plot_info([ax_fst,ax_last],["First spike","Last spike"])

plt.subplot(rows,columns,2)
ax1,ax_fst,ax_last =plot_events(laser_events,col=color_laser,tit=label2,width_ms=width,df_log=laser_log,show_durations=False)
set_plot_info([ax_fst,ax_last],["First spike","Last spike"])

plt.subplot(rows,columns,3)
ax1,ax_fst,ax_last =plot_events(control_pos_events,col=color_pos,tit=label3,width_ms=width,df_log=control_pos_log,show_durations=False)
set_plot_info([ax_fst,ax_last],["First spike","Last spike"])


#ControlPre-Laser

plt.subplot(rows,columns,4)
ax1,ax_fst,ax_last= plot_events(control_pre_events,color_pre,tit="ControlPre-Laser",width_ms=width)
ax2,ax_fst,ax_last=plot_events(laser_events,color_laser,tit="ControlPre-Laser",width_ms=width)

set_plot_info([ax1,ax2],[label1,label2],loc="lower left")


#ControlPos-Laser

plt.subplot(rows,columns,5)
ax1,ax_fst,ax_last= plot_events(control_pos_events,color_pos,tit="ControlPos-Laser",width_ms=width)
ax2,ax_fst,ax_last=plot_events(laser_events,color_laser,tit="ControlPos-Laser",width_ms=width)

set_plot_info([ax1,ax2],[label3,label2],loc="lower left")

#ControlPre-ControlPos

plt.subplot(rows,columns,6)
ax1,ax_fst,ax_last= plot_events(control_pre_events,color_pre,tit="ControlPre-ControlPos",width_ms=width)
ax3,ax_fst,ax_last=plot_events(control_pos_events,color_pos,tit="ControlPre-ControlPos",width_ms=width)

set_plot_info([ax1,ax3],[label1,label3],loc="lower left")

#Pre-Laser-Pos

plt.subplot(rows,columns,8)
ax1,ax_fst,ax_last= plot_events(control_pre_events,color_pre,tit="Pre-Laser-Pos",width_ms=width)
ax2,ax_fst,ax_last=plot_events(laser_events,color_laser,tit="Pre-Laser-Pos",width_ms=width)
ax3,ax_fst,ax_last= plot_events(control_pos_events,color_pos,tit="Pre-Laser-Pos",width_ms=width)

set_plot_info([ax1,ax2,ax3],[label1,label2,label3],loc="lower left")


plt.suptitle(path) #general title
plt.tight_layout(rect=[0, 0, 1, 0.95]) #tight with upper title
if save=='y':
	plt.savefig(path +".png")
if show=='y':
	plt.show()


if stats=='y':
	#Saving dataframes
	print("saving dataframes")

	df = create_dataframe([control_pre_log,laser_log,control_pos_log],['control_pre_','laser_','control_pos_'])
	print(df.describe())
	df.to_pickle(path+"_info.pkl")


# print("Duration differences")
# print("control to control:",df['control_pre_duration'].mean()-df['control_pos_duration'].mean())
# print("control pre to laser:",df['control_pre_duration'].mean()-df['laser_duration'].mean())
# print("control pos to laser:",df['control_pos_duration'].mean()-df['laser_duration'].mean())

# print("Amplitude differences")
# print("control to control:",df['control_pre_amplitude'].mean()-df['control_pos_amplitude'].mean())
# print("control pre to laser:",df['control_pre_amplitude'].mean()-df['laser_amplitude'].mean())
# print("control pos to laser:",df['control_pos_amplitude'].mean()-df['laser_amplitude'].mean())


# print("Slope inc differences")
# print("control to control:",df['control_pre_slope_dep'].mean()-df['control_pos_slope_dep'].mean())
# print("control pre to laser:",df['control_pre_slope_dep'].mean()-df['laser_slope_dep'].mean())
# print("control pos to laser:",df['control_pos_slope_dep'].mean()-df['laser_slope_dep'].mean())

# print("Slope dec differences")
# print("control to control:",df['control_pre_slope_rep'].mean()-df['control_pos_slope_rep'].mean())
# print("control pre to laser:",df['control_pre_slope_rep'].mean()-df['laser_slope_rep'].mean())
# print("control pos to laser:",df['control_pos_slope_rep'].mean()-df['laser_slope_rep'].mean())