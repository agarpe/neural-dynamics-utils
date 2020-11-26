# import pandas as pd 
# import sys
# import glob
# import os
# import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from stats_plot_functions import *


plt.rcParams.update({'font.size': 17})

def plot_barchart(df_dir,id_,labels,colors = ['b','r','g'],rows=4,cols=2):
	dur_means = df_dir[duration_labels].mean()
	amp_means = df_dir[amplitude_labels].mean()
	slo_dep_means = df_dir[slope_dep_labels].mean()
	slo_rep_means = df_dir[slope_rep_labels].mean()

	diff_labels=['control_pre-control_pos','control_pre-laser', 'control_pos-laser']
	indexes = [id_-0.15,id_,id_+0.15]

	plot_mean_bars(dur_means,labels,rows,cols,1,duration_title,duration_unit,colors,diff_labels,indexes=indexes)
	plot_mean_bars(amp_means,labels,rows,cols,3,amplitude_title,amplitude_unit,colors,diff_labels,indexes=indexes)
	plot_mean_bars(slo_dep_means,labels,rows,cols,5,slope_dep_title,slope_unit,colors,diff_labels,indexes=indexes)
	plot_mean_bars(slo_rep_means,labels,rows,cols,7,slope_rep_title,slope_unit,colors,diff_labels,indexes=indexes)



if len(sys.argv) ==3:
	path = sys.argv[1]
	extension = sys.argv[2]
elif len(sys.argv) ==2:
	path = sys.argv[1]
	extension = ""
else:
	print("Use: python3 stats_plot.py path")
	exit()

dirs = sorted(glob.glob(path+"*%s*"%extension))
# dirs.sort(key=os.path.getmtime)

print(dirs)

if(dirs==[]):
	print("Error: No dirs found. Check the extension provided")

plt.figure(figsize=(15,20))

labels=[]
for i,d in enumerate(dirs):
	# print(d)
	dir_name = d[d.rfind("/")+1:]
	print(dir_name)
	labels.append(dir_name)

	#ignore regular files
	if dir_name.find(".")!=-1:
		continue

	all_trials=[]
	files = glob.glob(d+"*/events/*.pkl")
	files.sort(key=os.path.getmtime)


	for j,f in enumerate(files):
		# print("\t",f)

		df = pd.read_pickle(f)
		# print(df.describe())

		df["Trial"]=j
		all_trials.append(df)

	all_trials=pd.concat(all_trials)

	plot_barchart(all_trials,i,labels)


print(labels)


plt.tight_layout()

plt.savefig(path+"general_barchart.eps",format="eps")
plt.show()
