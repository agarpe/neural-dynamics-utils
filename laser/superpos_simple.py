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

<<<<<<< HEAD
df = {}
ax1,ax_fst,ax_last=plot_events(events,col='b',tit=label,width_ms=width,df_log=df,show_durations=True)
=======
dur_log = []
slo_log = []
df = {}
ax1,ax_fst,ax_last=plot_events(events,col='b',tit=label,width_ms=width,duration_log=dur_log,df_log=df,amplitude_log=[],slope_log=slo_log,show_durations=True)
>>>>>>> bcc4c6d3f8293f6403b4c771de8d5c552b1a8cf2
plt.legend([ax_fst,ax_last],["First spike","Last spike"])
plt.xlabel("Time (ms)")
plt.ylabel("Voltage (mV)")


print(df)
df=pd.DataFrame(df)
# df = df.add_suffix('_control_pre')
print(df.describe())

print("saving dataframes")
<<<<<<< HEAD
df= pd.concat([df.add_prefix('control_pre_'),df.add_suffix('laser_')],axis=1)
=======
df= pd.concat([df.add_suffix('_control_pre'),df.add_suffix('_laser')],axis=1)
>>>>>>> bcc4c6d3f8293f6403b4c771de8d5c552b1a8cf2
print(df.describe())

# df.to_pickle(path+"_info.pkl")

# inc_slo, dec_slo = zip(*slo_log)
# inc_slo=np.array(inc_slo)
# dec_slo=np.array(dec_slo)


# data_tuples=list(itertools.zip_longest(inc_slo,dec_slo))
# df = pd.DataFrame(data_tuples,columns=["Increasing","Decreasing"])
# print(df.describe())

path = path[:path.find("exp")] +title

plt.suptitle(title)
plt.tight_layout(rect=[0, 0, 1, 0.95])
# plt.savefig(path +".png")
if(show):
	plt.show()
