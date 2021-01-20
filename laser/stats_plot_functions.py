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
	
	custom_lines=[Line2D([0], [0], color=colors[i], lw=4) for i in range(len(colors))]
	plt.legend(custom_lines,labels,fontsize=13)

def plot_mean_bars(means,labels,rows,cols,id_,title,unit,colors,indexes=[1,2,3],width=0.1,rotation=70,legends=['control_pre','laser','control_pos']):
	plt.subplot(rows,cols,id_)
	plt.bar(indexes,means,color=colors,width=width)
	plt.ylabel("Mean value %s"%unit)

	custom_lines=[Line2D([0], [0], color=colors[i], lw=4) for i in range(len(colors))]

	if legends!=[]:
		plt.legend(custom_lines,legends,fontsize=13)
	plt.title(title)
	plt.xticks(range(0,len(labels)),labels,rotation=rotation)

def plot_mean_n_diffs_bars(means,labels,rows,cols,id_,title,unit,colors,diff_labels,indexes=[1,2,3],width=0.1,rotation=70,legends=['control_pre','laser','control_pos']):
	plot_mean_bars(means,labels,rows,cols,id_,title,unit,colors,indexes,width,rotation,legends)

	plt.subplot(rows,cols,id_+1)
	plot_diffs(means,diff_labels,title,unit,colors,indexes=indexes)
	plt.xticks(range(0,len(labels)),labels,rotation=rotation)



#Function used for data of the type control1-laser-control2
def plot_barchart(df_dir,id_,labels,colors = ['b','r','g'],rows=4,cols=2):
	dur_means = df_dir[duration_labels].mean()
	amp_means = df_dir[amplitude_labels].mean()
	slo_dep_means = df_dir[slope_dep_labels].mean()
	slo_rep_means = df_dir[slope_rep_labels].mean()

	diff_labels=['control_pre-control_pos','control_pre-laser', 'control_pos-laser']
	indexes = [id_-0.15,id_,id_+0.15]

	plot_mean_n_diffs_bars(dur_means,labels,rows,cols,1,duration_title,duration_unit,colors,diff_labels,indexes=indexes)
	plot_mean_n_diffs_bars(amp_means,labels,rows,cols,3,amplitude_title,amplitude_unit,colors,diff_labels,indexes=indexes)
	plot_mean_n_diffs_bars(slo_dep_means,labels,rows,cols,5,slope_dep_title,slope_unit,colors,diff_labels,indexes=indexes)
	plot_mean_n_diffs_bars(slo_rep_means,labels,rows,cols,7,slope_rep_title,slope_unit,colors,diff_labels,indexes=indexes)

#Function generally used for models. 
def plot_barchart_simple(df_dir,id_,labels,colors = ['b','r','g'],rows=4,cols=1):
	dur_means = [df_dir["duration"].mean(),df_dir["duration"].std(),df_dir["duration"].min(),df_dir["duration"].max()]
	amp_means = [df_dir["amplitude"].mean(),df_dir["amplitude"].std(),df_dir["amplitude"].min(),df_dir["amplitude"].max()]
	slo_dep_means = [df_dir["slope_dep"].mean(),df_dir["slope_dep"].std(),df_dir["slope_dep"].min(),df_dir["slope_dep"].max()]
	slo_rep_means = [df_dir["slope_rep"].mean(),df_dir["slope_rep"].std(),df_dir["slope_rep"].min(),df_dir["slope_rep"].max()]

	legends=["mean","std","min","max"]
	indexes = [id_,id_+0.15,id_+0.15*2,id_+0.15*3]
	colors=['darkorange','darkcyan','lightgreen','darkolivegreen']


	plot_mean_bars(dur_means,labels,rows,cols,1,duration_title,duration_unit,colors,indexes=indexes,legends=legends)
	plot_mean_bars(amp_means,labels,rows,cols,2,amplitude_title,amplitude_unit,colors,indexes=indexes,legends=legends)
	plot_mean_bars(slo_dep_means,labels,rows,cols,3,slope_dep_title,slope_unit,colors,indexes=indexes,legends=legends)
	plot_mean_bars(slo_rep_means,labels,rows,cols,4,slope_rep_title,slope_unit,colors,indexes=indexes,legends=legends)



