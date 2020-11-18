import pandas as pd 
import sys
import glob
import os
import matplotlib.pyplot as plt

def set_titles(axes,title,path):
	for ax in axes.values():
		ax.set_ylabel(title)
	plt.suptitle(path)

def plot_boxplot(columns,title,path,fliers=True):
	axes=all_trials.boxplot(column=columns,by='Trial',grid=False,layout=(3,1),return_type='axes',figsize=(10,15),fontsize=20,showmeans=True,showfliers=fliers)
	set_titles(axes,title,path)


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

duration_labels = ['control_pre_duration','laser_duration','control_pos_duration']
amplitude_labels = ['control_pre_amplitude','laser_amplitude','control_pos_amplitude']


duration_title = 'Spike width (ms)'
amplitude_title = 'Spike amplitude (mV)'


plot_boxplot(duration_labels,duration_title,path)
plt.savefig(path +"duration_boxplots"+ extension+".png")

if show:	
	plt.show()
else:
	plt.clf()

plot_boxplot(duration_labels,duration_title,path,fliers=False)
plt.savefig(path +"duration_boxplots_no_fliers"+ extension+".png")

# 
if show:	
	plt.show()
else:
	plt.clf()


plot_boxplot(amplitude_labels,amplitude_title,path)
plt.savefig(path +"amplitude_boxplots"+ extension+".png")

# 
if show:	
	plt.show()
else:
	plt.clf()

plot_boxplot(amplitude_labels,amplitude_title,path,fliers=False)
plt.savefig(path +"amplitude_boxplots_no_fliers"+ extension+".png")

if show:	
	plt.show()
else:
	plt.clf()

dur_pre_mean = all_trials['control_pre'].mean()
dur_las_mean = all_trials['laser'].mean()
dur_pos_mean = all_trials['control_pos'].mean()

amp_pre_mean = all_trials['control_pre_amplitude'].mean()
amp_las_mean = all_trials['laser_amplitude'].mean()
amp_pos_mean = all_trials['control_pos_amplitude'].mean()


dur_means = [all_trials['control_pre'].mean(),all_trials['laser'].mean(),all_trials['control_pos'].mean()]
amp_means = [all_trials['control_pre_amplitude'].mean(),all_trials['laser_amplitude'].mean(),all_trials['control_pos_amplitude'].mean()]



plt.rcParams.update({'font.size': 25})


labels = ['control_pre','control_pre_amplitude','laser','laser_amplitude','control_pos','control_pos_amplitude']
plt.figure(figsize=(30,30))
plt.subplot(2,2,1)
plt.bar([1,2,3],dur_means,color='g',width=0.3,tick_label=duration_labels)
plt.ylabel("Mean value (ms)")
plt.title("Spike width")

plt.subplot(2,2,3)
plt.bar([1,2,3],amp_means,color='b',width=0.3,tick_label=duration_labels)
plt.ylabel("Mean value (mV)")
plt.title("Spike amplitude")

diffs_duration = [dur_pre_mean-dur_las_mean,dur_pre_mean-dur_pos_mean,dur_pos_mean-dur_las_mean]
diffs_amplitude = [amp_pre_mean-amp_las_mean,amp_pre_mean-amp_pos_mean,amp_pos_mean-amp_las_mean]

plt.subplot(2,2,2)
plt.bar([1,2,3],diffs_duration,tick_label=['control_pre-laser', 'control_pre-control_pos','control_pos-laser'],width=0.3,color='g')
plt.title("Spike width")
plt.ylabel("Difference value (ms)")
plt.subplot(2,2,4)
plt.bar([1,2,3],diffs_amplitude,tick_label=['control_pre-laser', 'control_pre-control_pos','control_pos-laser'],width=0.3,color='b')
plt.title("Spike amplitude")
plt.ylabel("Difference value (mV)")


plt.suptitle(path)
plt.savefig(path +"bar_chart"+ extension+".png")

if show:	
	plt.show()
else:
	plt.clf()