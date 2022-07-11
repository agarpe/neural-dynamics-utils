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
plt.rcParams.update({'font.size': 15})


import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=True, help="Path to the experiment trial")
ap.add_argument("-ws", "--window_width", required=True, help="Half window width in ms")
ap.add_argument("-sc", "--scale", required=False, default=1, help="Scale for Volts. Signal*scale")
ap.add_argument("-ti", "--title", required=False, default='', help="Title shown in plot")
ap.add_argument("-nr", "--rows", required=False, default=2, 
				help="Number of rows. 2 for single and comparations. 3 for prevs and single subplot with the three of them")
ap.add_argument("-co", "--color", required=False, default=0, 
				help="Color type. 0: light blue; red; green. 1: progressive lumniance of light blue; red; darkblue")
ap.add_argument("-ex", "--ext", required=False, default='', help="Extension after laser or control.")
ap.add_argument("-mean","--mean",required=False, default='n',help="When == 'y'. Plot mean of all spikes and not spike per spike.")
ap.add_argument("-ali","--align",required=False, default='peak',help="Choose alignment mode. 'peak', 'min', 'max', 'ini', 'end'")
ap.add_argument("-sa", "--save", required=False, default='y', help="Option to save plot file. 'y' or 'n'; default: y")
ap.add_argument("-sh", "--show", required=False, default='n', help="Option to show plot file. 'y' or 'n'; default: n")
ap.add_argument("-st", "--stats", required=False, default='y', help="Option to save stats pkl file. 'y' or 'n'; default: y")
ap.add_argument("-dir", "--dir", required=False, default='y', help="Save stats in new dir. 'y' or 'n'; default: y")
# ap.add_argument("-sp", "--stim_path", required=False, default=None, help="Path to the stimulation times")
ap.add_argument("-sr", "--stim_ref", required=False, default=True, help="Plot stimulus time references from *_laser_shutter_waveform_references.txt file")
ap.add_argument("-er", "--stim_error", required=False, default=np.inf, help="Allowed error in stimulation in ms")
ap.add_argument("-dt", "--time_step", required=False, default=0.1, help="Time step")

args = vars(ap.parse_args())

path = args['path']
ext = args['ext']

if ext != '':
	ext += '_'


# path_control_pre = path+"_control_pre_"+ext+"waveform.txt"
# path_laser = path+"_laser_"+ext+"waveform.txt"
# path_control_pos = path+"_control_pos_"+ext+"waveform.txt"

path_control_pre = path+"_control_"+ext+"waveform_single.txt"
path_laser = path+"_laser_"+ext+"waveform_single.txt"
path_control_pos = path+"_recovery_"+ext+"waveform_single.txt"

color = int(args['color'])

width = int(args['window_width'])
# TODO add into args and functions 
width_l = width
width_r = width

rows = int(args['rows'])
scale = float(args['scale'])
dt = float(args['time_step'])

mode = args['align']

show= True if args['show']=='y' else False 
save= True if args['save']=='y' else False 
stats= True if args['stats']=='y' else False 
in_dir= True if args['dir']=='y' else False 

stim = True if args['stim_ref']=='y' else False 
# if stim is not None:
if stim:
	try:
		# stim_path = path + '_laser_shutter_waveform_references.txt' # name from v1
		stim_path = path + '_laser_shutter_time_references.txt'
		stim = np.loadtxt(stim_path)
		print("Stim ",stim.shape)
	except Exception as e:
		stim = False
		print("EXCEPTION:",e.args)
		pass

error = float(args['stim_error'])


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
	cols = (width_l+width_r)/dt + 100
	print (cols)
	control_pre_events = read_from_events(path_control_pre,max_cols=cols,dt=0.1)
	laser_events =  read_from_events(path_laser,max_cols=cols,dt=0.1)
	control_pos_events =  read_from_events(path_control_pos,max_cols=cols,dt=0.1)
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
control_pre_events = control_pre_events.values*scale
laser_events = laser_events.values*scale
control_pos_events = control_pos_events.values*scale

control_pre_events = control_pre_events[2:-2]
laser_events = laser_events[2:-2]
control_pos_events = control_pos_events[2:-2]


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

if stim is not False:
	if 'depol' in path or 'slope' in path:
		# error = 5
		laser_events, ids = preprocess_spikes(laser_events, stim[:,1], width_l=width, error=error)
		stim = stim[ids]

	elif 'repol' in path:
		# error = 5
		laser_events, ids = preprocess_spikes(laser_events, stim[:,0], width_l=width, error=error)
		stim = stim[ids]


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
	ax1,ax_fst,ax_last =plot_func(control_pre_events,col=color_pre,tit=label1,width_ms_l=width, width_ms_r=width,df_log=control_pre_log,show_durations=False, mode=mode)
	set_plot_info([ax_fst,ax_last],legends,width)

	plt.subplot(rows,columns,2)
	ax1,ax_fst,ax_last =plot_func(laser_events,col=color_laser,tit=label2,width_ms_l=width, width_ms_r=width,df_log=laser_log,show_durations=False, mode=mode)
	set_plot_info([ax_fst,ax_last],legends,width)

	plt.subplot(rows,columns,3)
	ax1,ax_fst,ax_last =plot_func(control_pos_events,col=color_pos,tit=label3,width_ms_l=width, width_ms_r=width,df_log=control_pos_log,show_durations=False, mode=mode)
	set_plot_info([ax_fst,ax_last],legends,width)


	#ControlPre-Laser

	plt.subplot(rows,columns,4)
	ax1,ax_fst,ax_last= plot_func(control_pre_events,color_pre,tit=label1+"-"+label2,width_ms_l=width, width_ms_r=width, mode=mode)
	ax2,ax_fst,ax_last=plot_func(laser_events,color_laser,tit=label1+"-"+label2,width_ms_l=width, width_ms_r=width, mode=mode)

	set_plot_info([ax1,ax2],[label1,label2],width,loc="lower left")


	#ControlPos-Laser

	plt.subplot(rows,columns,5)
	ax1,ax_fst,ax_last= plot_func(control_pos_events,color_pos,tit=label3+"-"+label2,width_ms_l=width, width_ms_r=width, mode=mode)
	ax2,ax_fst,ax_last=plot_func(laser_events,color_laser,tit=label3+"-"+label2,width_ms_l=width, width_ms_r=width, mode=mode)

	set_plot_info([ax1,ax2],[label3,label2],width,loc="lower left")

	#ControlPre-ControlPos

	plt.subplot(rows,columns,6)
	ax1,ax_fst,ax_last= plot_func(control_pre_events,color_pre,tit=label1+"-"+label3,width_ms_l=width, width_ms_r=width, mode=mode)
	ax3,ax_fst,ax_last=plot_func(control_pos_events,color_pos,tit=label1+"-"+label3,width_ms_l=width, width_ms_r=width, mode=mode)

	set_plot_info([ax1,ax3],[label1,label3],width,loc="lower left")

	#Pre-Laser-Pos
	if rows == 3:
		plt.subplot(rows,columns,8)

if rows != 2:
	if rows == 1:
		plt.figure(figsize=(10,rows*10))
	ax1,ax_fst,ax_last= plot_func(control_pre_events,color_pre,tit=label1+"-"+label2+"-"+label3,width_ms_l=width, width_ms_r=width, mode=mode)
	ax2,ax_fst,ax_last=plot_func(laser_events,color_laser,tit=label1+"-"+label2+"-"+label3,width_ms_l=width, width_ms_r=width, mode=mode)
	ax3,ax_fst,ax_last= plot_func(control_pos_events,color_pos,tit=label1+"-"+label2+"-"+label3,width_ms_l=width, width_ms_r=width, mode=mode)

	xlim = 'default'

	if stim is not False:
		xlim = None

		# remove rows full of Nan
		laser_events = laser_events[(~np.isnan(laser_events).all(axis=1))]
		# max events in ms
		
		#TODO: fix el /10 no tiene sentido pero no sale bien sino ...
		maxs = np.nanargmax(laser_events, axis=1) * dt /10
		# print(maxs)
		# # print(maxs)
		# print(width_l)
		# print(stim)
		# print(width_l*dt)
		# inis = np.array([max_id - s[0] + (width_l - max_id) for s, max_id in zip(stim, maxs)])
		# ends = np.array([max_id + s[1] + (width_r - max_id) for s, max_id in zip(stim, maxs)])

		#TODO: no siempre es +/- max_id
		# if 'repol' in path:
		# 	inis = np.array([max_id - s[0] + (width_l - max_id) for s, max_id in zip(stim, maxs)])
		# 	ends = np.array([max_id + s[1] + (width_r - max_id) for s, max_id in zip(stim, maxs)])
		# else:
		# 	inis = np.array([max_id - s[0] + (width_l - max_id) for s, max_id in zip(stim, maxs)])
		# 	ends = np.array([max_id - s[1] + (width_r - max_id) for s, max_id in zip(stim, maxs)])

		inis = np.array([max_id - s[0] for s, max_id in zip(stim, maxs)])
		ends = np.array([max_id - s[1] for s, max_id in zip(stim, maxs)])

		# plt.plot(maxs, np.ones(maxs.shape),'.')
		plt.plot(inis, np.zeros(inis.shape),'x',color='k',label='ini')
		plt.plot(ends, np.zeros(ends.shape),'|',color='k',label='end')
		plt.legend()

	set_plot_info([ax1,ax2,ax3],[label1,label2,label3],width,loc="center right",xlim=xlim)


title = '\n'.join (title.split('\\'))

plt.suptitle(title,wrap=True) #general title
plt.tight_layout(rect=[0, 0, 1, 0.95]) #tight with upper title

if save:
	if  args['title'] == '':
		title=''
	if args['mean'] == 'y':
		m = 'mean_'
	else:
		m = ''

	if stim is not False:
		stim_str = '_stim_error_%f'%error
	else:
		stim_str = ''

	path_images = path[:path.rfind("/")] + '/images/' + mode + path[path.rfind('/'):] 
	print(path_images)
	figname=path_images +"_" + ext + title + m + str(width) + "_" + str(args['rows']) + "_" + mode + stim_str

	os.system("mkdir -p %s"%path_images[:path_images.rfind('/')])

	print("Saving plot at",figname)
	plt.savefig(figname+".png")
	if(args['mean'] == 'y'):
		plt.savefig(figname+".eps",format='eps',dpi=1200)
	# plt.savefig(figname+".pdf",format='pdf',dpi=600)
if show:
	plt.figure()
	# # plt.hist(maxs) center before ploting
	plt.hist(stim[:,0],width=10,label='inits')
	plt.hist(stim[:,1],width=10,label='ends')
	plt.legend()
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
