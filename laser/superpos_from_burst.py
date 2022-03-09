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

plt.rcParams.update({'font.size': 30})


import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=True, help="Path to the experiment trial")
ap.add_argument("-ws", "--window_width", required=True, help="Half window width in ms")
ap.add_argument("-sc", "--scale", required=False, default=10, help="Scale for Volts. Signal*scale")
ap.add_argument("-ti", "--title", required=False, default='', help="Title shown in plot")
ap.add_argument("-nr", "--rows", required=False, default=2, 
				help="Number of rows. 2 for single and comparations. 3 for prevs and single subplot with the three of them")
ap.add_argument("-ex", "--ext", required=False, default='', help="Extension after laser or control.")
ap.add_argument("-sa", "--save", required=False, default='y', help="Option to save plot file")
ap.add_argument("-sh", "--show", required=False, default='y', help="Option to show plot file")
ap.add_argument("-st", "--stats", required=False, default='y', help="Option to save stats pkl file")
args = vars(ap.parse_args())


path = args['path']
ext = args['ext']

if ext != '':
	ext += '_'

path_control_pre = path

width = int(args['window_width'])
scale = int(args['scale'])
rows = int(args['rows'])

show= True if args['show']=='y' else False 
save= True if args['save']=='y' else False 
stats= True if args['stats']=='y' else False 

if args['title'] == '':
	title = path
else:
	title = args['title']
print(title)

print("\nSuperposing from ",path)

print("Reading events files...")

try:
	#Each row contains Voltage values of the corresponding event.
	control_pre_events = read_from_events(path_control_pre,max_cols=1000,dt=0.1)
except:
	print("Error: file not found")
	print(path_control_pre)
	exit()


n_control_pre = len(control_pre_events.index)


#Parse to array
control_pre_events=control_pre_events.values*scale


#Labels for Control-Laser
label1 = "Control pre"
label1_nspikes = label1+". N. spikes: %d"%(n_control_pre)


#Dafaframes and logs
control_pre_log ={}

color_pre = 'b'

#Error: invalid color???
# color_pre = (Color("lightcyan"),Color("cornflowerblue"))
# color_pos = (Color("skyblue"),Color("darkblue"))
# color_laser = (Color("lightsalmon"),Color("darkred"))



# blue = Color("blue")
# color = blue
# color.luminance = luminances[i%(len(files))]
# color = color.hex_l
colors = {'b':['cyan','darkblue'],'r':['coral','maroon'],'g':['lime','darkgreen']}

#------------------------------------------------

#########################################################
######## Plot in grid ################################
########################################################
# rows = 3 
# rows = 2
# columns= 3
# plt.figure(figsize=(columns*10,rows*10))

# f = plot_events
plot_f = burst_plot

ax1,ax_fst,ax_last= plot_f(control_pre_events,color_pre,tit="Burst superpos",width_ms=width)
set_plot_info([ax1], [label1], width, loc="lower left", xlim=None)


plt.suptitle(title) #general title
plt.tight_layout(rect=[0, 0, 1, 0.95]) #tight with upper title

if save:
	if  args['title'] == '':
		title=''
	figname=path +"_"+ext+"_"+title
	plt.savefig(figname+".png")
	# plt.savefig(figname+".eps",format='eps',dpi=1200)
	plt.savefig(figname+".pdf",format='pdf',dpi=600)
if show:
	plt.show()

