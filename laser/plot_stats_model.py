# Author: Alicia Garrido-Peña
# Date: 30-01-2023

import pickle as pkl
from matplotlib.lines import Line2D
# from stats_plot_functions import *
# from superpos_from_model import get_first_line
import glob
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys

import superpos_functions as laser_utils
script_path = sys.argv[0][:sys.argv[0].rfind('/')]
sys.path.append(script_path+'/..')
import plot_utils as pu


def get_first_line(file):
	try:
		if 'pkl' in file:
			with open(file[:-3]+'info','r') as fs:
				first_line = fs.readline()

			# print(first_line)
		else:
			# fs=open(file)
			with open(file,'r') as fs:
				first_line = fs.readline()
			# fs.close()

		if first_line == '':
			raise Exception
	except Exception as e:
		print(e)
		print("Skiped",file)
		first_line = ''
		# raise Exception
	# index =fs.find(f_format)
	# ini = fs.rfind("_")

	return first_line

def plot_grid_by_metric():

	# print(all_df['Cm'])
	n_rows = 2
	n_cols = 2
	print(n_cols)
	fig,axis = plt.subplots(figsize=(25,8),nrows = n_rows, ncols=n_cols)

	key_colors = plt.cm.tab20(np.linspace(0,1,len(all_df.keys())))
	key_colors = np.flip(key_colors, axis=0)

	for rc,(key,value) in enumerate(zip(all_df.keys(), all_df.values())):
		boxprops = dict(linestyle='-', linewidth=2, color=key_colors[rc])
		
		for i,metric in enumerate(metrics):
			value.boxplot(column=metric, ax=axis[i//2,i%2], positions=[rc], showfliers=False, showmeans=True, grid=False, fontsize=20, boxprops=boxprops)
		
			axis[i//2,i%2].set_title(metric)


	for ai in axis:
		for a in ai:
			ax=a.axes
			ax.set_xticklabels([list(all_df.keys())[n] for n in range(len(ax.get_xticklabels()))])

	plt.tight_layout()	




import argparse

# 	print("Example: python3 superpos_from_model.py ../../laser_model/HH/data/gna/ gna \"Gna simulation\" 0.001 8 20")
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=True, help="Path to the file to show stats from")
ap.add_argument("-pe", "--path_extension", required=False,default="", help="Path extension to the files to show stats from")
ap.add_argument("-sa", "--save", required=False, default='y', help="Option to save plot file")
ap.add_argument("-sh", "--show", required=False, default='y', help="Option to show plot file")
ap.add_argument("-dt", "--time_step", required=False, default=0.01, help="Sampling freq of -fs")

args = vars(ap.parse_args())


path = args['path']
ext_path = args['path_extension'] #name of the parameter varied during simulations
show= True if args['show']=='y' else False 
save= True if args['save']=='y' else False 

dt = float(args['time_step'])

dirs = sorted(glob.glob(path+"/*/"))
# dirs.sort(key=os.path.getmtime)

print(dirs)


if(dirs==[]):
	print("Error: No dirs found. Check the extension provided")


path = path.replace('*','')
name = path.replace('/','')

all_df = {}

df = pd.DataFrame()
for i,dir_ in enumerate(dirs):
	files = sorted(glob.glob(dir_+"/*waveform.pkl"))

	param = dir_[dir_.find('/')+2:-1]
	print(param)

	# if 'cgc' in param:
	# 	continue
	plt.figure()
	plt.title(dir_)

	for file in files:
		# print(file)
		with open(file,'rb') as f:
			waveform = pkl.load(f)

		# print(waveform.shape)
		# plt.plot(waveform.T)
		# plt.show()

		ref = file.find("Euler")
		if(ref!=-1):
			f_events = file[:ref]+"spikes_"+file[ref:file.rfind('waveform')-1]+'.pkl'
		else:
			f_events = file[:file.rfind('waveform')]+"spikes.pkl"#+f_format


		try:
			first_line = get_first_line(f_events)
		except Exception as e:
			print("************************",e)
			continue
		if first_line == '':
			continue

		# print(first_line)

		if 'HH' in file or 'Vav' in file:
			dt = 0.001
			n_points=3
			repol_points=10
		else:
			dt = 0.01
			n_points=10
			repol_points=60
		# print(dt)

		dur_refs,th = laser_utils.get_spike_duration(waveform, dt)
		duration = dur_refs[1]-dur_refs[0]
		# print("Duration value:", duration)

		amplitude = laser_utils.get_spike_amplitude(waveform, dt)
		# print("Amplitude value:", amplitude)

		slope_dep, slope_rep = laser_utils.get_slope(waveform, dt, n_points=n_points, plot=True)
		# slopes_dep2, slopes_rep2 = get_slopes(sf.get_slope2, waveforms, slope_position=slope_position)
		slopes_dep2, slopes_rep2 = laser_utils.get_slope2(waveform, dt, n_points=n_points, repol_points=repol_points, plot=True)


		if param == 'Cm':
			param = 'Cm\n'+dir_[:dir_.find('/')]

		new_row = pd.DataFrame([{'param':float(first_line), 'duration':duration, 'amplitude':amplitude, 'slope_dep':slope_dep, 'slope_rep':slope_rep}])

		try:
			all_df[param] = pd.concat([all_df[param], new_row])
		except:
			all_df[param] = pd.DataFrame()
			all_df[param] = pd.concat([all_df[param], new_row])
	
	if show:
		plt.show()

# exit()
print(all_df.keys())

plt.rcParams.update({'font.size': 20})

metrics = ['duration','slope_dep','slope_rep','amplitude']

print(all_df)
print(all_df.keys())

# Get normalized differences
df_diffs = {}
for rc,(key,value) in enumerate(zip(all_df.keys(), all_df.values())):
	#Value is the dataframe for each Candidate/Neuron (directory in the path given)
	# value.iloc[:,:] = value.iloc[:,:].apply(lambda x: (abs(x).max()-abs(x).min())/abs(x).max(), axis=0)
	# max_ = value.loc[value['param'] == value['param'].max()]
	# min_ = value.loc[value['param'] == value['param'].min()]
	max_ = value.abs().max()
	min_ = value.abs().min()
	df_diffs[key] = (abs(max_)-abs(min_))/abs(max_)


pretty_metrics = ['Duration', 'Depolarization', 'Repolarization', 'Amplitude']
shoulder_values = [0.43,0.28,0.86,0.015]
symmetric_values = [0.24,0.11,0.26,0.028]

colors = {'HH':'#ffc28cff','CGC':'#dcc857ae','N3t':'#ccbaa1ff'}

def plot_bar(data):
	fig,ax = plt.subplots(nrows=len(metrics),figsize=(1.8*len(data.keys()),9))

	# for rc,(key,value) in enumerate(zip(all_df.keys(), all_df.values())):
	for i, metric in enumerate(metrics):
		# plot reference line and area from shoulder and symmetrical
		ax[i].axhspan(max(shoulder_values[i]-0.05,0),min(shoulder_values[i]+0.05,1), facecolor='g', alpha=0.1)
		ax[i].axhspan(max(symmetric_values[i]-0.05,0),min(symmetric_values[i]+0.05,1), facecolor='g', alpha=0.1)
		ax[i].axhline(y=shoulder_values[i], linestyle='-.', color='g', label='shoulder' )
		ax[i].axhline(y=symmetric_values[i], linestyle='--', color='g', label='symmetrical' )
		ax[0].legend(loc='upper right', fontsize=11)

		for rc,(key,value) in enumerate(zip(data.keys(), data.values())):
		#Value is the dataframe for each Candidate/Neuron (directory in the path given)
			try:
				ckeys = [ckey for ckey in colors.keys() if ckey in key]
				# print(ckeys)
				if any(ckeys):
					color = colors[ckeys[0]]
			except:
				color = 'C0'

			# print(color)

			ax[i].bar(rc*0.22, value[metric], color=color, alpha=0.8, width=0.2)

			# if i==1:
			# ax[i].set_ylabel(u"Δ%s"%pretty_metrics[i])
			ax[i].set_yticks([])

			ax[i].set_xticklabels(data.keys())
			ax[i].set_xticks(np.arange(len(data.keys()))*0.22)
			ax[i].set_ylim(0,1)

			pu.remove_axes(ax[i])
		if i < len(metrics)-1:
			ax[i].set_xticklabels([])

	plt.subplots_adjust(wspace=0, hspace=0)

	plt.suptitle(path)
	plt.tight_layout()

	if save:
		plt.savefig(path+'/'+name+"_general_bar.svg",format='svg')

plot_bar(df_diffs)


if show:
	plt.show()