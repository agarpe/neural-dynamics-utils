import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
from superpos_functions import *
import itertools

plt.rcParams.update({'font.size': 17})


if len(sys.argv) ==6:
	path = sys.argv[1]
	width = int(sys.argv[2])
	label= sys.argv[3]
	color= sys.argv[4]
	title=sys.argv[5]
	show = True
else:
	print("Use1: python3 superpos_from_events.py events_path.txt width label color title")
	exit()

os.system("sed -i 's/\,/./g' "+path) #changing , to . to read floats not strings

# #Each row contains voltage values of the corresponding event.
events = read_from_events(path,max_cols=300,dt=0.1)

n_events = len(events.index)

# #Remove last column NaN values
# events=events.iloc[:, :-1] 

#Parse to array
events=events.values

print(events.shape)

# #Labels for Control-Laser
label = label+" "+str(n_events)

#------------------------------------------------
# Plot 

plt.figure(figsize=(20,15))
plt.tight_layout()

#Individual plots

dur_log = []
slo_log = []

ax1,ax_fst,ax_last=plot_events(events,col='b',tit=label,width_ms=width,duration_log=dur_log,slope_log=slo_log,show_durations=True)
plt.legend([ax_fst,ax_last],["First spike","Last spike"])
plt.xlabel("Time (ms)")
plt.ylabel("Voltage (mV)")

list1, list2 = zip(*slo_log)
list1=np.array(list1)
list2=np.array(list2)


data_tuples=list(itertools.zip_longest(list1,list2))
df = pd.DataFrame(data_tuples,columns=["Increasing","Decreasing"])
print(df.describe())

path = path[:path.find("exp")] +title

plt.suptitle(title)
plt.tight_layout(rect=[0, 0, 1, 0.95])
# plt.savefig(path +".png")
if(show):
	plt.show()
