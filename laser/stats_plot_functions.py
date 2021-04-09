import pandas as pd 
import sys
import glob
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

legend_fontsize = 25
ticks_fontsize = 30

n_spikes_labels = ['control_pre_count','laser_count','control_pos_count']
duration_labels = ['control_pre_duration','laser_duration','control_pos_duration']
amplitude_labels = ['control_pre_amplitude','laser_amplitude','control_pos_amplitude']
slope_dep_labels = ['control_pre_slope_dep','laser_slope_dep','control_pos_slope_dep']
slope_rep_labels = ['control_pre_slope_rep','laser_slope_rep','control_pos_slope_rep']

n_spikes_labels = ['count']
duration_labels = ['duration']
amplitude_labels = ['amplitude']
slope_dep_labels = ['slope_dep']
slope_rep_labels = ['slope_rep']

n_spike_title = 'Number of spikes'; n_spikes_unit = ''
duration_title = 'Spike duration'; duration_unit = '(ms)'
amplitude_title = 'Spike amplitude'; amplitude_unit = '(mV)'
slope_dep_title = 'Depolarization slope'; slope_unit = ''
slope_rep_title = 'Repolarization slope'; slope_unit = ''

titles = {'spikes':{'labels':n_spikes_labels,'title':n_spike_title},'duration':{'labels':duration_labels,'title':duration_title},
'amplitude':{'labels':amplitude_labels,'title':amplitude_title},'slope_dep':{'labels':slope_dep_labels,'title':slope_dep_title},
'slope_rep':{'labels':slope_rep_labels,'title':slope_rep_title}}

# titles_model ={'spikes':{'labels':n_spikes_label,'title':n_spike_title},'duration':{'labels':duration_label,'title':duration_title},
# 'amplitude':{'labels':amplitude_label,'title':amplitude_title},'slope_dep':{'labels':slope_dep_label,'title':slope_dep_title},
# 'slope_rep':{'labels':slope_rep_label,'title':slope_rep_title}}

def plot_boxplot(df,columns,title,path,fliers=True,rot_val=0):
	lay=(len(columns),1)
	figsize=(10,len(columns)*5)
	axes=df.boxplot(column=columns,by='Trial',grid=False,layout=lay,return_type='axes',figsize=figsize,fontsize=20,showmeans=True,showfliers=fliers,rot=rot_val)
	for ax in axes.values():
	# for ax in axes:
		ax.set_ylabel(title)
	plt.suptitle(path)
	plt.tight_layout()

def plot_mean_bars(means,errors,labels,fig_size,id_,title,unit,colors,indexes=[1,2,3],width=0.1,rotation=60,legends=['control_pre','laser','control_pos'],error_kw=dict(lw=1, capsize=4, capthick=1.5)):
	plt.subplot(fig_size[1],fig_size[0],id_)
	plt.bar(indexes,means,yerr=errors,color=colors,width=width,error_kw=error_kw)
	plt.ylabel("Mean value %s"%unit)

	custom_lines=[Line2D([0], [0], color=colors[i], lw=4) for i in range(len(colors))]

	if legends!=[]:
		plt.legend(custom_lines,legends,fontsize=legend_fontsize)
	plt.title(title)
	plt.xticks(range(0,len(labels)),labels,rotation=rotation,horizontalalignment='right',fontsize=ticks_fontsize)


def get_diffs(means):
	diffs = [abs(means[0]-means[2]),abs(means[0]-means[1]),abs(means[2]-means[1])]
	return diffs

def plot_diffs(means,errors,labels,title,unit,colors,indexes=[1,2,3],width=0.1):
	diffs=get_diffs(means)
	plt.bar(indexes,diffs,yerr=errors,width=width,color=colors)
	plt.title(title)
	plt.ylabel("Difference value %s"%unit)
	
	custom_lines=[Line2D([0], [0], color=colors[i], lw=4) for i in range(len(colors))]
	plt.legend(custom_lines,labels,fontsize=legend_fontsize)

def plot_mean_n_diffs_bars(means,errors,labels,fig_size,id_,title,unit,colors,indexes=[1,2,3],width=0.1,rotation=60,legends=['control_pre','laser','control_pos'],error_kw=dict(lw=1, capsize=4, capthick=1.5)):
	plot_mean_bars(means,errors,labels[0],fig_size,id_,title,unit,colors,indexes,width,rotation,legends)

	plt.subplot(fig_size[1],fig_size[0],id_+1)
	plot_diffs(means,errors,labels[1],title,unit,colors,indexes=indexes,width=width)
	plt.xticks(range(0,len(labels)),labels,rotation=rotation,horizontalalignment='right',fontsize=ticks_fontsize)


#Function used for data of the type control1-laser-control2
# df_dir: dataframe containing (at least) the columns specified in columns arg.
# columns: list of strings containing the columns to plot. 
# 			p.e. ['spikes','duration','amplitude','slope_rep','slope_dep']
# id_ : index of dataframe to be ploted used to set bars position in the plot. 
# labels: list of strings with ticks labels.
# colors: list with strings with bars colors 
# rows: number of rows for the subplots
# cols: number of columns for the subplot
# plot_diffs: if True function plot_mean_n_diffs_bars is called, computing the difference
#			  of 3 data sets. p.e control_pre, laser, control_pos.
# error_kw: error bar plot set parameters
# titles: dict with titles associated to the columns. Containing for each column: labels, title, unit.
#
#	Warning: this function might be called several times for same plot (id_ argument).
#			The tick_labels in the plot will keep last value.
def plot_barchart(df_dir,id_,labels,columns,colors,fig_size,
	plot_diffs=False,error_kw=dict(lw=1, capsize=4, capthick=1.5),
	titles=titles,legends=['control_pre','laser','control_pos']):
	# rows = len(columns)
	# print(df_dir,id_,labels,colors,rows,cols,plot_diffs,columns,error_kw,titles)
	for i,col in enumerate(columns):
		means = df_dir[titles[col]['labels']].mean()
		error = df_dir[titles[col]['labels']].std()

		if legends != []:
			indexes = [id_-0.23,id_,id_+0.23]
		else:
			indexes = id_

		if(plot_diffs):
			diff_labels=['control_pre-control_pos','control_pre-laser', 'control_pos-laser']
			plot_f = plot_mean_n_diffs_bars
			labels = [labels,diff_labels]
			plot_id = i*2+1
		else:
			plot_f = plot_mean_bars
			plot_id = i+1

		plot_f(means,error,labels,fig_size,plot_id,titles[col]['title'],n_spikes_unit,colors,indexes=indexes,width=0.2,error_kw=error_kw,legends=legends)



# #Function generally used for models. 
# def plot_barchart_simple(df_dir,id_,labels,colors = ['b','r','g'],rows=4,cols=1):
# 	dur_means = [df_dir["duration"].mean(),df_dir["duration"].std(),df_dir["duration"].min(),df_dir["duration"].max()]
# 	amp_means = [df_dir["amplitude"].mean(),df_dir["amplitude"].std(),df_dir["amplitude"].min(),df_dir["amplitude"].max()]
# 	slo_dep_means = [df_dir["slope_dep"].mean(),df_dir["slope_dep"].std(),df_dir["slope_dep"].min(),df_dir["slope_dep"].max()]
# 	slo_rep_means = [df_dir["slope_rep"].mean(),df_dir["slope_rep"].std(),df_dir["slope_rep"].min(),df_dir["slope_rep"].max()]

# 	legends=["mean","std","min","max"]
# 	indexes = [id_,id_+0.15,id_+0.15*2,id_+0.15*3]
# 	colors=['darkorange','darkcyan','lightgreen','darkolivegreen','green']


# 	plot_mean_bars(dur_means,labels,rows,cols,1,duration_title,duration_unit,colors,indexes=indexes,legends=legends)
# 	plot_mean_bars(amp_means,labels,rows,cols,2,amplitude_title,amplitude_unit,colors,indexes=indexes,legends=legends)
# 	plot_mean_bars(slo_dep_means,labels,rows,cols,3,slope_dep_title,slope_unit,colors,indexes=indexes,legends=legends)
# 	plot_mean_bars(slo_rep_means,labels,rows,cols,4,slope_rep_title,slope_unit,colors,indexes=indexes,legends=legends)


# #Function generally used for models. 
# def plot_barchart_simple(df_dir,id_,labels,colors = ['b','r','g'],rows=4,cols=1):
# 	dur_means = [df_dir["duration"].mean(),df_dir["duration"].std(),df_dir["duration"].min(),df_dir["duration"].max(),df_dir["duration"].max()-df_dir["duration"].min()]
# 	amp_means = [df_dir["amplitude"].mean(),df_dir["amplitude"].std(),df_dir["amplitude"].min(),df_dir["amplitude"].max(),df_dir["amplitude"].max()-df_dir["amplitude"].min()]
# 	slo_dep_means = [df_dir["slope_dep"].mean(),df_dir["slope_dep"].std(),df_dir["slope_dep"].min(),df_dir["slope_dep"].max(),df_dir["slope_dep"].max()-df_dir["slope_dep"].min()]
# 	slo_rep_means = [df_dir["slope_rep"].mean(),df_dir["slope_rep"].std(),df_dir["slope_rep"].min(),df_dir["slope_rep"].max(),df_dir["slope_rep"].max()-df_dir["slope_rep"].min()]

# 	legends=["mean","std","min","max","diff-width"]
# 	indexes = [id_,id_+0.15,id_+0.15*2,id_+0.15*3,id_+0.15*4]
# 	colors=['darkorange','darkcyan','lightgreen','darkolivegreen','green']


# 	plot_mean_bars(dur_means,labels,rows,cols,1,duration_title,duration_unit,colors,indexes=indexes,legends=legends)
# 	plot_mean_bars(amp_means,labels,rows,cols,2,amplitude_title,amplitude_unit,colors,indexes=indexes,legends=legends)
# 	plot_mean_bars(slo_dep_means,labels,rows,cols,3,slope_dep_title,slope_unit,colors,indexes=indexes,legends=legends)
# 	plot_mean_bars(slo_rep_means,labels,rows,cols,4,slope_rep_title,slope_unit,colors,indexes=indexes,legends=legends)





