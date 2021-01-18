import pandas as pd 
import sys
import glob
import os
import matplotlib.pyplot as plt
from stats_plot_functions import *


import argparse

# 	print("Example: python3 superpos_from_model.py ../../laser_model/HH/data/gna/ gna \"Gna simulation\" 0.001 8 20")
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=True, help="Path to the file to show stats from")
ap.add_argument("-pe", "--path_extension", required=False,default="", help="Path extension to the files to show stats from")
ap.add_argument("-sa", "--save", required=False, default='y', help="Option to save plot file")
ap.add_argument("-sh", "--show", required=False, default='y', help="Option to show plot file")
ap.add_argument("-fl", "--fliers", required=False, default='n', help="Option to show boxplot fliers")
args = vars(ap.parse_args())


path = args['path']
extension = args['path_extension'] #name of the parameter varied during simulations
show= True if args['show']=='y' else False 
save= True if args['save']=='y' else False 
fliers= True if args['fliers']=='y' else False 



files = glob.glob(path+"*%s*.pkl"%extension)
files.sort(key=os.path.getmtime)


all_trials=[]
labels=[]
print(files)
print(path)

if(files==[]):
	print("Error: No files found. Check the extension provided")
	exit()

plt.figure(figsize=(30,35))
for i,f in enumerate(files):
	# print(f)
	df = pd.read_pickle(f)
	# print(df.describe())

	name = f[f.rfind("/")+1:]
	# print(name)
	val = name[name.rfind("-")+1:name.rfind("_")]
	df["Trial"]=val
	all_trials.append(df)

	# plt.figure(figsize=(30,35))
	labels.append(val)
	plot_barchart_simple(df,i,labels)

# plt.tight_layout()
plt.savefig(path +"general_barchart"+ extension+".eps",format="eps")
# plt.savefig(path +"general_barchart"+ extension+".png",format="png")
# plt.show()

all_trials=pd.concat(all_trials)

duration_labels = ['duration']
amplitude_labels = ['amplitude']
slope_dep_labels = ['slope_dep']
slope_rep_labels = ['slope_rep']


# print(df.describe())

plot_boxplot(all_trials,duration_labels,duration_title+duration_unit,path,rot_val=-90)
if save:
	plt.savefig(path +"duration_boxplots"+ extension+".png")
if fliers:
	plot_boxplot(all_trials,duration_labels,duration_title+duration_unit,path,fliers=False,rot_val=-90)
	if save:
		plt.savefig(path +"duration_boxplots_no_fliers"+ extension+".png")

plot_boxplot(all_trials,amplitude_labels,amplitude_title+amplitude_unit,path,rot_val=-90)
if save:
	plt.savefig(path +"amplitude_boxplots"+ extension+".png")
if fliers:
	plot_boxplot(all_trials,amplitude_labels,amplitude_title+amplitude_unit,path,fliers=False,rot_val=-90)
	if save:
		plt.savefig(path +"amplitude_boxplots_no_fliers"+ extension+".png")


plot_boxplot(all_trials,slope_dep_labels,slope_dep_title,path)
if save:
	plt.savefig(path +"slope_dep_boxplots"+ extension+".png")
if fliers:
	plot_boxplot(all_trials,slope_dep_labels,slope_dep_title,path,fliers=False,rot_val=-90)
	if save:
		plt.savefig(path +"slope_dep_boxplots_no_fliers"+ extension+".png")

plot_boxplot(all_trials,slope_rep_labels,slope_rep_title,path,rot_val=-90)
if save:
	plt.savefig(path +"slope_rep_boxplots"+ extension+".png")
if fliers:
	plot_boxplot(all_trials,slope_rep_labels,slope_rep_title,path,fliers=False,rot_val=-90)
	if save:
		plt.savefig(path +"slope_rep_boxplots_no_fliers"+ extension+".png")

# plt.rcParams.update({'font.size': 30})


if save:
	plt.savefig(path +"general_barchart"+ extension+".png")

if show:	
	plt.show()
