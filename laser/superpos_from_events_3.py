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

# plt.rcParams.update({'font.size': 30})
plt.rcParams.update({'font.size': 25})


import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=True, help="Path to the experiment trial")
ap.add_argument("-ws", "--window_width", required=True, help="Half window width in ms")
ap.add_argument("-sc", "--scale", required=False, default=1, help="Scale for Volts. Signal*scale")
ap.add_argument("-ti", "--title", required=False, default='', help="Title shown in plot")
ap.add_argument("-nr", "--rows", required=False, default=2, 
				help="Number of rows. 2 for single and comparations. 3 for prevs and single subplot with the three of them")
ap.add_argument("-co", "--color", required=False, default=0, 
				help="Color type. 0: light blue; red; darkblue. 1: progressive lumniance of light blue; red; darkblue")
ap.add_argument("-ex", "--ext", required=False, default='', help="Extension after laser or control.")
ap.add_argument("-mean","--mean",required=False, default='n',help="When == 'y'. Plot mean of all spikes and not spike per spike.")
ap.add_argument("-sa", "--save", required=False, default='y', help="Option to save plot file")
ap.add_argument("-sh", "--show", required=False, default='y', help="Option to show plot file")
ap.add_argument("-st", "--stats", required=False, default='y', help="Option to save stats pkl file")
ap.add_argument("-dir", "--dir", required=False, default='y', help="Save stats in new dir")
args = vars(ap.parse_args())

path = args['path']
ext = args['ext']

if ext != '':
	ext += '_'

path_control_pre = path+"_control_pre_"+ext+"waveform.txt"
path_laser = path+"_laser_"+ext+"waveform.txt"
path_control_pos = path+"_control_pos_"+ext+"waveform.txt"

color = int(args['color'])

width = int(args['window_width'])
rows = int(args['rows'])
scale = float(args['scale'])

show= True if args['show']=='y' else False 
save= True if args['save']=='y' else False 
stats= True if args['stats']=='y' else False 
in_dir= True if args['dir']=='y' else False 

plot_func = plot_events_mean if args['mean']=='y' else plot_events 

if args['title'] == '':
	title = path
else:
	title = args['title']
print(title)

print("\nSuperposing from ",path)

print("Reading events files...")

control_pre_events = df_ = pd.DataFrame(index=[], columns=[])
laser_events = pd.DataFrame(index=[], columns=[])
control_pos_events = pd.DataFrame(index=[], columns=[])

try:
	#Each row contains Voltage values of the corresponding event.
	control_pre_events = read_from_events(path_control_pre,max_cols=300,dt=0.1)
	laser_events =  read_from_events(path_laser,max_cols=300,dt=0.1)
	control_pos_events =  read_from_events(path_control_pos,max_cols=300,dt=0.1)
except FileNotFoundError as e:
	print("Error: one of the files could not be found")
	print(e)
except Exception as e:
	print("Error: ")
	print(e)
	# print(path_control_pre)
	# exit()


n_control_pre = len(control_pre_events.index)
n_laser = len(laser_events.index)
n_control_pos = len(control_pos_events.index)


#Parse to array
control_pre_events=control_pre_events.values*scale
laser_events=laser_events.values*scale
control_pos_events=control_pos_events.values*scale


#Labels for Control-Laser
label1 = "Control pre"
label2 = "Laser"
label3 = "Recovery"
# label3 = "Control pos"


#Dafaframes and logs
control_pre_log ={}
laser_log ={}
control_pos_log ={}

if color ==0:
	color_pre = 'b'
	color_laser = 'r'
	color_pos = 'g'
else:
	color_pre = (Color("lightcyan"),Color("cornflowerblue"))
	color_pos = (Color("skyblue"),Color("darkblue"))
	color_laser = (Color("lightsalmon"),Color("darkred"))

# colors = {'b':['cyan','darkblue'],'r':['coral','maroon'],'g':['lime','darkgreen']}

#------------------------------------------------
if rows >1:
	#########################################################
	######## Plot in grid ################################
	########################################################
	# rows = 3 
	# rows = 2
	columns= 3
	plt.figure(figsize=(columns*10,rows*10))



	#Individual plots
	if plot_func==plot_events:
		legends = ["First spike","Last spike"]
	else:
		legends = []
	plt.subplot(rows,columns,1)
	ax1,ax_fst,ax_last =plot_func(control_pre_events,col=color_pre,tit=label1,width_ms=width,df_log=control_pre_log,show_durations=False)
	set_plot_info([ax_fst,ax_last],legends,width)

	plt.subplot(rows,columns,2)
	ax1,ax_fst,ax_last =plot_func(laser_events,col=color_laser,tit=label2,width_ms=width,df_log=laser_log,show_durations=False)
	set_plot_info([ax_fst,ax_last],legends,width)

	plt.subplot(rows,columns,3)
	ax1,ax_fst,ax_last =plot_func(control_pos_events,col=color_pos,tit=label3,width_ms=width,df_log=control_pos_log,show_durations=False)
	set_plot_info([ax_fst,ax_last],legends,width)


	#ControlPre-Laser

	plt.subplot(rows,columns,4)
	ax1,ax_fst,ax_last= plot_func(control_pre_events,color_pre,tit=label1+"-"+label2,width_ms=width)
	ax2,ax_fst,ax_last=plot_func(laser_events,color_laser,tit=label1+"-"+label2,width_ms=width)

	set_plot_info([ax1,ax2],[label1,label2],width,loc="lower left")


	#ControlPos-Laser

	plt.subplot(rows,columns,5)
	ax1,ax_fst,ax_last= plot_func(control_pos_events,color_pos,tit=label3+"-"+label2,width_ms=width)
	ax2,ax_fst,ax_last=plot_func(laser_events,color_laser,tit=label3+"-"+label2,width_ms=width)

	set_plot_info([ax1,ax2],[label3,label2],width,loc="lower left")

	#ControlPre-ControlPos

	plt.subplot(rows,columns,6)
	ax1,ax_fst,ax_last= plot_func(control_pre_events,color_pre,tit=label1+"-"+label3,width_ms=width)
	ax3,ax_fst,ax_last=plot_func(control_pos_events,color_pos,tit=label1+"-"+label3,width_ms=width)

	set_plot_info([ax1,ax3],[label1,label3],width,loc="lower left")

	#Pre-Laser-Pos
	if rows == 3:
		plt.subplot(rows,columns,8)

if rows != 2:
	if rows == 1:
		plt.figure(figsize=(10,rows*10))
	ax1,ax_fst,ax_last= plot_func(control_pre_events,color_pre,tit=label1+"-"+label2+"-"+label3,width_ms=width)
	ax2,ax_fst,ax_last=plot_func(laser_events,color_laser,tit=label1+"-"+label2+"-"+label3,width_ms=width)
	ax3,ax_fst,ax_last= plot_func(control_pos_events,color_pos,tit=label1+"-"+label2+"-"+label3,width_ms=width)

	set_plot_info([ax1,ax2,ax3],[label1,label2,label3],width,loc="center left")


plt.suptitle(title) #general title
plt.tight_layout(rect=[0, 0, 1, 0.95]) #tight with upper title

if save:
	if  args['title'] == '':
		title=''
	figname=path +"_"+ext+title
	plt.savefig(figname+".png")
	plt.savefig(figname+".eps",format='eps',dpi=1200)
	plt.savefig(figname+".pdf",format='pdf',dpi=600)
if show:
	plt.show()

if stats:
	if control_pre_log == {} or control_pos_log == {} or laser_log == {}:
		print("Stats were not saved. No log info created")
		exit()
	#Saving dataframes
	print("saving dataframes")

	df = create_dataframe([control_pre_log,laser_log,control_pos_log],['control_pre_','laser_','control_pos_'])
	print(df.describe())

	if not in_dir:
		df.to_pickle(path+"_"+ext+"info.pkl")
	
	else:
		#saves same df in dir
		os.system("mkdir -p %s"%path+"_"+ext)
		indx = path.rfind("/")
		df.to_pickle(path[:indx+1]+path[indx:]+"_"+ext+"/"+path[indx+1:]+"_"+ext+"info.pkl")

