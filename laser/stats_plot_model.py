import pandas as pd 
import sys
import glob
import os
import matplotlib.pyplot as plt
from stats_plot_functions import *

if len(sys.argv) ==3:
	path = sys.argv[1]
	extension = sys.argv[2]
elif len(sys.argv) ==2:
	path = sys.argv[1]
	extension = ""
else:
	print("Use: python3 stats_plot.py path")
	exit()

show = True
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
	# print(df.describe())

	name = f[f.rfind("/")+1:]
	print(name)
	val = name[name.rfind("-")+1:name.rfind("_")]
	df["Trial"]=val
	all_trials.append(df)


all_trials=pd.concat(all_trials)

duration_labels = ['duration']
amplitude_labels = ['amplitude']
slope_dep_labels = ['slope_dep']
slope_rep_labels = ['slope_rep']


print(df.describe())

plot_boxplot(all_trials,duration_labels,duration_title+duration_unit,path,rot_val=-90)
if save:
	plt.savefig(path +"duration_boxplots"+ extension+".png")
plot_boxplot(all_trials,duration_labels,duration_title+duration_unit,path,fliers=False,rot_val=-90)
if save:
	plt.savefig(path +"duration_boxplots_no_fliers"+ extension+".png")

plot_boxplot(all_trials,amplitude_labels,amplitude_title+amplitude_unit,path,rot_val=-90)
if save:
	plt.savefig(path +"amplitude_boxplots"+ extension+".png")
plot_boxplot(all_trials,amplitude_labels,amplitude_title+amplitude_unit,path,fliers=False,rot_val=-90)
if save:
	plt.savefig(path +"amplitude_boxplots_no_fliers"+ extension+".png")


plot_boxplot(all_trials,slope_dep_labels,slope_dep_title,path)
if save:
	plt.savefig(path +"slope_dep_boxplots"+ extension+".png")
plot_boxplot(all_trials,slope_dep_labels,slope_dep_title,path,fliers=False,rot_val=-90)
if save:
	plt.savefig(path +"slope_dep_boxplots_no_fliers"+ extension+".png")

plot_boxplot(all_trials,slope_rep_labels,slope_rep_title,path,rot_val=-90)
if save:
	plt.savefig(path +"slope_rep_boxplots"+ extension+".png")
plot_boxplot(all_trials,slope_rep_labels,slope_rep_title,path,fliers=False,rot_val=-90)
if save:
	plt.savefig(path +"slope_rep_boxplots_no_fliers"+ extension+".png")

# plt.rcParams.update({'font.size': 30})


if show:	
	plt.show()
