import pandas as pd 
import sys
import glob
import os
import matplotlib.pyplot as plt

if len(sys.argv) ==2:
	path = sys.argv[1]
else:
	print("Use: python3 analyze_amplitudes.py path")
	exit()

files = glob.glob(path+"*.pkl")
files.sort(key=os.path.getmtime)


all_trials=[]
print(files)

for i,f in enumerate(files):
	print(f)
	df = pd.read_pickle(f)
	print(df.describe())

	df["Trial"]=i
	all_trials.append(df)


all_trials=pd.concat(all_trials)

axes=all_trials.boxplot(column=['control_pre','laser','control_pos'],by='Trial',grid=False,layout=(3,1),return_type='axes',figsize=(10,15),fontsize=20)
plt.ylabel("Spike width (ms)")
for ax in axes.values():
	ax.set_ylabel("Spike width (ms)")
# plt.show()

plt.savefig(path +"boxplots.png")
