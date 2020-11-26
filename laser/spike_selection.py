import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os,sys,inspect
import argparse
sys.path.append('..')
import charact_utils as utils
import scipy.stats as stats

from superpos_functions import *

matplotlib.rcParams.update({'font.size': 16})

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=True, help="Path to the file to analyze")
ap.add_argument("-pw", "--path_waveforms", required=True, help="Path to the file to analyze")
ap.add_argument("-mxz", "--z_value", required=True, help="Max Z value")
ap.add_argument("-s", "--show", required=False,default=True, help="Show plot")
args = vars(ap.parse_args())

path = args['path']
path_wf = args['path_waveforms']
mx_z = float(args['z_value'])
if(args['show']=='False'):
	show = False
else:
	show = True


print(path)
print(mx_z)
spikes = utils.read_spike_events(path)


print(spikes.shape)


#open spike waveforms
spikes_wf = read_from_events(path_wf)
spikes_wf=spikes_wf.values[:-1]
print(spikes_wf.shape)




isis = np.array(utils.get_ISI(spikes))
print(isis.shape)

zscores = stats.zscore(isis, axis=0, ddof=0)

# plt.hist(isis,width=1)
# plt.hist(zscores,width=0.1,bins=10)
# plt.show()
# plt.hist(zscores[np.where(zscores<0)],width=0.1,bins=6)



# width_ms = 10000
# dt=1
# width_points = width_ms /dt #Number of points corresponding to the iteration
# sigma = 1000
# sdf_ = utils.sdf(spikes,spike_window=200,window_size=width_points,sigma=sigma)

# plt.plot(sdf_)
# plt.plot(spikes[1:],isis,'.')
# plt.plot(spikes[1:],zscores,'.')
# plt.show()

zscores=abs(zscores)

to_keep = []
for i,z in enumerate(zscores):
	if(z >= mx_z):
		to_keep.append(i)

# to_keep = np.where((isis>500) & (isis <1000))[0]
to_keep = np.where((isis>100) & (isis <1000))[0]

print("Spikes to remove: ",len(list(set(to_keep))))
try:
	# spikes_select = np.delete(spikes,to_keep) 
	spikes_select = spikes[to_keep]
	#remove spikes
	# spikes_wf_sel = np.delete(spikes_wf,to_keep,axis=0) 
	spikes_wf_sel = spikes_wf[to_keep] 
	zscores_select = zscores[to_keep]
	# zscores_select = np.delete(zscores,to_keep) 

except:
	print("index failed: ",i)


plt.figure(figsize=(20,15))
ax1=plt.subplot(3,1,1)
plt.plot(spikes,np.ones(spikes.shape[0]),'.')
plt.plot(spikes_select,np.ones(spikes_select.shape[0]),'.','r')
plt.subplot(3,1,2, sharex = ax1)
# plt.plot(spikes[:-1],zscores,'.')
# plt.plot(spikes_select[:-1],zscores_select,'.')
plt.subplot(3,1,3, sharex = ax1)
# plt.subplot(212)
# plt.plot(spikes[:-1],isis,'.')
# plt.plot(spikes_select[:-1],np.delete(isis,to_keep),'.')
plt.show()


print(spikes_wf_sel.shape)

#Plot result
plt.figure(figsize=(10,15))
rows = 2
columns= 1

plt.subplot(rows,columns,2)
ax1,ax_fst,ax_last =plot_events(spikes_wf_sel,col='b',tit='spike selection',width_ms=50)
set_plot_info([ax_fst,ax_last],["First spike","Last spike"])


plt.subplot(rows,columns,1)
ax1,ax_fst,ax_last =plot_events(spikes_wf,col='g',tit='spikes',width_ms=50)
set_plot_info([ax_fst,ax_last],["First spike","Last spike"])


# plt.savefig(path[:-3]+"png")
if show:
	plt.show()
