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

	for rc,(key,value) in enumerate(all_df.items()):
		boxprops = dict(linestyle='-', linewidth=2, color=key_colors[rc])
		
		for i,metric in enumerate(metrics):
			value.boxplot(column=metric, ax=axis[i//2,i%2], positions=[rc], showfliers=False, showmeans=True, grid=False, fontsize=20, boxprops=boxprops)
		
			axis[i//2,i%2].set_title(metric)


	for ai in axis:
		for a in ai:
			ax=a.axes
			ax.set_xticklabels([list(all_df.keys())[n] for n in range(len(ax.get_xticklabels()))])

	plt.tight_layout()	


def generate_table(df):
	import seaborn as sns


	# Create a custom colormap that transitions from red to palegreen and back to red
	colors = sns.color_palette(["salmon", "mediumseagreen", "salmon"])
	colors = sns.color_palette(["salmon", "mediumseagreen","mediumseagreen", "salmon"])
	colors = sns.color_palette(["darkblue", "white", "white", "darkblue"])
	cm1 = sns.blend_palette(colors, as_cmap=True)

	fig,ax = plt.subplots()
	pu.plot_cmap(fig, ax, colors, location=[0.5,0,0.08,1]) # [x, y, width, height]
	fig.delaxes(ax) 
	plt.savefig('color_bar'+'.pdf', format='pdf', bbox_inches='tight')


	# Create a custom colormap that transitions from red to palegreen and back to red
	# colors = sns.color_palette(["salmon","mediumseagreen","mediumseagreen", "mediumseagreen", "salmon"])
	colors = sns.color_palette(["salmon","mediumseagreen", "salmon"])
	cm2 = sns.blend_palette(colors, as_cmap=True)

	# inverse map
	# colors = sns.color_palette(["mediumseagreen", "mediumseagreen", "salmon"])
	colors = sns.color_palette(["mediumseagreen", "mediumseagreen", "mediumseagreen", "salmon"])
	colors = sns.color_palette(["white", "darkblue", "darkblue",'white'])

	cm3 = sns.blend_palette(colors, as_cmap=True)

	# Desaturate colors
	# cm1 = pu.modify_cmap(cm1, desat=0.75, alpha=1, hue=1)
	# cm2 = pu.modify_cmap(cm2, desat=0.75, alpha=1, hue=1)
	# cm3 = pu.modify_cmap(cm3, desat=0.75, alpha=1, hue=1)


	# Create style object
	styler = df.style

	shoulder_values = [0.43,0.28,0.86,0.015]
	symmetric_values = [0.24,0.11,0.26,0.028]
	mins = [min(a,b) for a,b in zip(shoulder_values,symmetric_values)]
	maxs = [max(a,b) for a,b in zip(shoulder_values,symmetric_values)]
	print(mins, maxs)
	ranges = [(min_-min_*0.6,max_*1.4) for min_,max_ in zip(mins,maxs)]
	print(ranges)
	print()
	# Apply background color gradient to each column based on min-max values.
	# styler = styler.background_gradient(cmap=cm1, subset=['duration'], vmin=0, vmax=0.9)
	# styler = styler.background_gradient(cmap=cm1, subset=['depol.'], vmin=-0.1, vmax=0.8)
	
	# styler = styler.background_gradient(cmap=cm2, subset=['repol.'], vmin=-0.1, vmax=2)

	# styler = styler.background_gradient(cmap=cm3, subset=['amplitude'], vmin=0.03, vmax=0.15)
	styler = styler.background_gradient(cmap=cm1, subset=['duration'], vmin=ranges[0][0], vmax=ranges[0][1])
	styler = styler.background_gradient(cmap=cm1, subset=['depol.'], vmin=ranges[1][0], vmax=ranges[1][1])
	
	styler = styler.background_gradient(cmap=cm1, subset=['repol.'], vmin=ranges[2][0], vmax=ranges[2][1])

	styler = styler.background_gradient(cmap=cm3, subset=['amplitude'], vmin=ranges[3][0], vmax=ranges[3][1])

	# text style
	styler = styler.set_properties(**{'text-align': 'center', 'font-family': 'garuda','width': '100px'})
	styler = styler.format(precision=3)

	# Convert style object to HTML
	html = styler.to_html()

	# Save pdf with the table
	import pdfkit
	pdfkit.from_string(html, 'styled_table-%s.pdf'%name)
	

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

dirs = sorted(glob.glob(path+"/*/*"))
# dirs.sort(key=os.path.getmtime)

print(dirs)


if(dirs==[]):
	print("Error: No dirs found. Check the extension provided")


path = path.replace('*','')
name = path.replace('/','')

all_df = {}

df = pd.DataFrame()

# for c in candidate:
# 	print(c)
for i,dir_ in enumerate(dirs):
	files = sorted(glob.glob(dir_+"/*waveform.pkl"))

	param = dir_[dir_.find('/')+2:]

	param = ''.join([p for p in param if not p.isnumeric()])
	param = param.replace('/','_')
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

		dur_refs,th = laser_utils.get_spike_duration(waveform, dt, plot=True)
		duration = dur_refs[1]-dur_refs[0]
		# print("Duration value:", duration)

		amplitude = laser_utils.get_spike_amplitude(waveform, dt)
		# print("Amplitude value:", amplitude)

		slope_dep, slope_rep = laser_utils.get_slope(waveform, dt, n_points=n_points, plot=True)
		# slopes_dep2, slopes_rep2 = get_slopes(sf.get_slope2, waveforms, slope_position=slope_position)
		slopes_dep2, slopes_rep2 = laser_utils.get_slope2(waveform, dt, n_points=n_points, repol_points=repol_points, plot=True)


		if param == 'Cm':
			param = 'Cm\n'+dir_[:dir_.find('/')]

		new_row = pd.DataFrame([{'param':float(first_line), 'duration':duration, 'amplitude':amplitude, 'depol.':slope_dep, 'repol.':slope_rep}])

		try:
			all_df[param] = pd.concat([all_df[param], new_row])
		except:
			all_df[param] = pd.DataFrame()
			all_df[param] = pd.concat([all_df[param], new_row])
	
	# if show:
	# 	plt.show()
	plt.close()

# exit()
# print(all_df.keys())

plt.rcParams.update({'font.size': 20})

metrics = ['duration','depol.','repol.','amplitude']

# Get normalized differences
df_diffs = {}
for rc,(key,value) in enumerate(all_df.items()):
	#Value is the dataframe for each Candidate/Neuron (directory in the path given)
	max_ = value.max()
	min_ = value.min()

	df_diffs[key] = abs((max_ - min_))/abs(max_)

# plot table 
# get dataframe from dict
DF_diffs = pd.concat(df_diffs, axis=1).T
generate_table(DF_diffs[metrics])
# generate table colored by metric
