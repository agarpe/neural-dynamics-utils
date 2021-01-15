import pandas as pd 
import sys
import glob
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

duration_labels = ['control_pre_duration','laser_duration','control_pos_duration']
amplitude_labels = ['control_pre_amplitude','laser_amplitude','control_pos_amplitude']
slope_dep_labels = ['control_pre_slope_dep','laser_slope_dep','control_pos_slope_dep']
slope_rep_labels = ['control_pre_slope_rep','laser_slope_rep','control_pos_slope_rep']


duration_title = 'Spike width'; duration_unit = '(ms)'
amplitude_title = 'Spike amplitude'; amplitude_unit = '(mV)'
slope_dep_title = 'Depolarization slope'; slope_unit = ''
slope_rep_title = 'Repolarization slope'; slope_unit = ''


def plot_boxplot(df,columns,title,path,fliers=True,rot_val=0):
	lay=(len(columns),1)
	figsize=(10,len(columns)*5)
	axes=df.boxplot(column=columns,by='Trial',grid=False,layout=lay,return_type='axes',figsize=figsize,fontsize=20,showmeans=True,showfliers=fliers,rot=rot_val)
	for ax in axes.values():
		ax.set_ylabel(title)
	plt.suptitle(path)
	plt.tight_layout()

def get_diffs(means):
	diffs = [abs(means[0]-means[2]),abs(means[0]-means[1]),abs(means[2]-means[1])]
	return diffs

def plot_diffs(means,labels,title,unit,colors,indexes=[1,2,3],width=0.1):
	diffs=get_diffs(means)
	plt.bar(indexes,diffs,width=width,color=colors)
	plt.title(title)
	plt.ylabel("Difference value %s"%unit)
	custom_lines = [Line2D([0], [0], color=colors[0], lw=4),
	                Line2D([0], [0], color=colors[1], lw=4),
	                Line2D([0], [0], color=colors[2], lw=4)]
	plt.legend(custom_lines,labels,fontsize=13)

def plot_mean_bars(means,labels,rows,cols,id_,title,unit,colors,indexes=[1,2,3],width=0.1,rotation=70,legends=['control_pre','laser','control_pos']):
	plt.subplot(rows,cols,id_)
	plt.bar(indexes,means,color=colors,width=width)
	plt.ylabel("Mean value %s"%unit)
	custom_lines = [Line2D([0], [0], color=colors[0], lw=4),
	                Line2D([0], [0], color=colors[1], lw=4),
	                Line2D([0], [0], color=colors[2], lw=4)]
	if legend!=[]:
		plt.legend(custom_lines,legends,fontsize=13)
	plt.title(title)
	plt.xticks(range(0,len(labels)),labels,rotation=rotation)

def plot_mean_n_diffs_bars(means,labels,rows,cols,id_,title,unit,colors,diff_labels,indexes=[1,2,3],width=0.1,rotation=70,legends=['control_pre','laser','control_pos']):
	plot_mean_bars(means,labels,rows,cols,id_,title,unit,colors,indexes,width,rotation,legends)

	plt.subplot(rows,cols,id_+1)
	plot_diffs(means,diff_labels,title,unit,colors,indexes=indexes)
	plt.xticks(range(0,len(labels)),labels,rotation=rotation)
