import pandas as pd 
import sys
import glob
import os
import matplotlib.pyplot as plt

if len(sys.argv) ==3:
	path = sys.argv[1]
	extension = sys.argv[2]
elif len(sys.argv) ==2:
	path = sys.argv[1]
	extension = ""
else:
	print("Use: python3 analyze_amplitudes.py path")
	exit()

show = False

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

axes=all_trials.boxplot(column=['control_pre','laser','control_pos'],by='Trial',grid=False,layout=(3,1),return_type='axes',figsize=(10,15),fontsize=20,showmeans=True)
plt.ylabel("Spike width (ms)")
for ax in axes.values():
	ax.set_ylabel("Spike width (ms)")
plt.suptitle(path)

plt.savefig(path +"duration_boxplots"+ extension+".png")

if show:	
	plt.show()
else:
	plt.clf()


axes=all_trials.boxplot(column=['control_pre','laser','control_pos'],by='Trial',grid=False,layout=(3,1),return_type='axes',figsize=(10,15),fontsize=20,showmeans=True,showfliers=False)
plt.ylabel("Spike width (ms)")
for ax in axes.values():
	ax.set_ylabel("Spike width (ms)")
plt.suptitle(path)

plt.savefig(path +"duration_boxplots_no_fliers"+ extension+".png")

# 
if show:	
	plt.show()
else:
	plt.clf()



axes=all_trials.boxplot(column=['control_pre_amplitude','laser_amplitude','control_pos_amplitude'],by='Trial',grid=False,layout=(3,1),return_type='axes',figsize=(10,15),fontsize=20,showmeans=True)
plt.ylabel("Spike amplitude (mV)")
for ax in axes.values():
	ax.set_ylabel("Spike amplitude (mV)")
plt.suptitle(path)

plt.savefig(path +"amplitude_boxplots"+ extension+".png")

# 
if show:	
	plt.show()
else:
	plt.clf()


axes=all_trials.boxplot(column=['control_pre_amplitude','laser_amplitude','control_pos_amplitude'],by='Trial',grid=False,layout=(3,1),return_type='axes',figsize=(10,15),fontsize=20,showmeans=True,showfliers=False)
plt.ylabel("Spike amplitude (mV)")
for ax in axes.values():
	ax.set_ylabel("Spike amplitude (mV)")
plt.suptitle(path)

plt.savefig(path +"amplitude_boxplots_no_fliers"+ extension+".png")

# 
if show:	
	plt.show()
else:
	plt.clf()

print("\n\n\nMEANS")

df = all_trials.groupby(['Trial']).mean()
print(df.describe())
# df=df.sort_index()
# print(df.describe())
df.plot.bar()

# if show:	
# 	plt.show()
# else:
# 	plt.clf()
# plt.show()

df = all_trials.groupby(['control_pre']).mean()
print(df.describe())
# df=df.sort_index()
# print(df.describe())
df.plot.bar()

plt.show()

# if show:	
# 	plt.show()
# else:
# 	plt.clf()