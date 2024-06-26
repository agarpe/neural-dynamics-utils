import pandas as pd 
import sys
import glob
import os
import matplotlib.pyplot as plt
from stats_plot_functions import *

# #Function used for data of the type control1-laser-control2
# def plot_barchart(df_dir,id_,labels,colors = ['b','r','g'],rows=4,cols=2):
# 	dur_means = df_dir[duration_labels].mean()
# 	amp_means = df_dir[amplitude_labels].mean()
# 	slo_dep_means = df_dir[slope_dep_labels].mean()
# 	slo_rep_means = df_dir[slope_rep_labels].mean()

# 	diff_labels=['control_pre-control_pos','control_pre-laser', 'control_pos-laser']
# 	indexes = [id_-0.15,id_,id_+0.15]

# 	plot_mean_n_diffs_bars(dur_means,labels,rows,cols,1,duration_title,duration_unit,colors,diff_labels,indexes=indexes)
# 	plot_mean_n_diffs_bars(amp_means,labels,rows,cols,3,amplitude_title,amplitude_unit,colors,diff_labels,indexes=indexes)
# 	plot_mean_n_diffs_bars(slo_dep_means,labels,rows,cols,5,slope_dep_title,slope_unit,colors,diff_labels,indexes=indexes)
# 	plot_mean_n_diffs_bars(slo_rep_means,labels,rows,cols,7,slope_rep_title,slope_unit,colors,diff_labels,indexes=indexes)


if len(sys.argv) ==3:
	path = sys.argv[1]
	extension = sys.argv[2]
elif len(sys.argv) ==2:
	path = sys.argv[1]
	extension = ""
else:
	print("Use: python3 stats_plot.py path")
	exit()

show = False
save = True

files = glob.glob(path+"*%s*.pkl"%extension)
files.sort(key=os.path.getmtime)


all_trials=[]
print(files)

if(files==[]):
	print("Error: No files found. Check the extension provided")
	exit()

for i,f in enumerate(files):
	print(f)
	df = pd.read_pickle(f)
	print(df.describe())

	df["Trial"]=i
	all_trials.append(df)


all_trials=pd.concat(all_trials)



plot_boxplot(all_trials,duration_labels,duration_title+duration_unit,path)
if save:
	plt.savefig(path +"duration_boxplots"+ extension+".png")
plot_boxplot(all_trials,duration_labels,duration_title+duration_unit,path,fliers=False)
if save:
	plt.savefig(path +"duration_boxplots_no_fliers"+ extension+".png")

plot_boxplot(all_trials,amplitude_labels,amplitude_title+amplitude_unit,path)
if save:
	plt.savefig(path +"amplitude_boxplots"+ extension+".png")
plot_boxplot(all_trials,amplitude_labels,amplitude_title+amplitude_unit,path,fliers=False)
if save:
	plt.savefig(path +"amplitude_boxplots_no_fliers"+ extension+".png")


plot_boxplot(all_trials,slope_dep_labels,slope_dep_title,path)
if save:
	plt.savefig(path +"slope_dep_boxplots"+ extension+".png")
plot_boxplot(all_trials,slope_dep_labels,slope_dep_title,path,fliers=False)
if save:
	plt.savefig(path +"slope_dep_boxplots_no_fliers"+ extension+".png")

plot_boxplot(all_trials,slope_rep_labels,slope_rep_title,path)
if save:
	plt.savefig(path +"slope_rep_boxplots"+ extension+".png")
plot_boxplot(all_trials,slope_rep_labels,slope_rep_title,path,fliers=False)
if save:
	plt.savefig(path +"slope_rep_boxplots_no_fliers"+ extension+".png")

plt.rcParams.update({'font.size': 30})
rows = 4
cols = 2



plt.figure(figsize=(30,35))
plot_barchart(all_trials,1,[''])



# dur_means = all_trials[duration_labels].mean()
# amp_means = all_trials[amplitude_labels].mean()
# slo_dep_means = all_trials[slope_dep_labels].mean()
# slo_rep_means = all_trials[slope_rep_labels].mean()

# diff_labels=['control_pre-laser', 'control_pre-control_pos','control_pos-laser']
# colors=['b','r','g']
# plot_mean_n_diffs_bars(dur_means,duration_labels,rows,cols,1,duration_title,duration_unit,colors,diff_labels)
# plot_mean_n_diffs_bars(amp_means,amplitude_labels,rows,cols,3,amplitude_title,amplitude_unit,colors,diff_labels)
# plot_mean_n_diffs_bars(slo_dep_means,slope_dep_labels,rows,cols,5,slope_dep_title,slope_unit,colors,diff_labels)
# plot_mean_n_diffs_bars(slo_rep_means,slope_rep_labels,rows,cols,7,slope_rep_title,slope_unit,colors,diff_labels)

plt.suptitle(path)
plt.tight_layout()
if save:
	plt.savefig(path +"bar_chart"+ extension+".png")

if show:	
	plt.show()
