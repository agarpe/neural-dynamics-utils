import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os,sys,inspect
import argparse
sys.path.append('..')
import charact_utils as utils
import scipy.stats as stats
import statistics

from superpos_functions import *

matplotlib.rcParams.update({'font.size': 16})


def save_files(spikes_select,spikes_wf_sel,path,path_ext,file_name):
	indx=path.rfind("/")

	#path_ext: directory with events usually "events"
	save_path_events = path[:indx]+"/"+path_ext+path[indx:]+"%s_events.txt"%file_name
	save_path_waveforms = path[:indx]+"/"+path_ext+path[indx:]+"%s_waveform.txt"%file_name

	print("Saving events data at \t\n",save_path_events)
	try:
		utils.save_events(spikes_select,save_path_events,split=True)
		print("Saving waveform data at \t\n",save_path_waveforms)
		waveforms_f = open(save_path_waveforms,'w')
		np.savetxt(waveforms_f,spikes_wf_sel,delimiter='\t')
	except FileNotFoundError:
		print("Could not save file. Path provided not found")
		exit()



ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=True, help="Path to the file to analyze")
ap.add_argument("-ps", "--path_save", required=True, help="Path to save new file to analyze")
ap.add_argument("-fn", "--file_name", required=True, help="File name on orifinal path to analyze")
ap.add_argument("-fe", "--file_extension", required=True, help="New file selection extension")
ap.add_argument("-sh", "--show", required=False,default='n', help="Show plot")
ap.add_argument("-sa", "--save", required=False,default='n', help="Save results")
args = vars(ap.parse_args())

path = args['path'] #../../data/laser/Jul-27-2020/events/
file_name = args['file_name']

path_events = path+file_name+"_events.txt"
path_wf = path+file_name+"_waveform.txt"

file_ext = args['file_extension']
path_save = args['path_save']

show= True if args['show']=='y' else False 
save= True if args['save']=='y' else False 


print(path)
print(path_events)
print(path_wf)
spikes = utils.read_spike_events(path_events)


print(spikes.shape)


#open spike waveforms
spikes_wf = read_from_events(path_wf)
spikes_wf=spikes_wf.values[:-1]
print(spikes_wf.shape)




isis = np.array(utils.get_ISI(spikes))
print(isis.shape)


durs =[]

for spike in spikes_wf:
	spike = spike[~np.isnan(spike)]
	dur = get_spike_duration(spike,0.1)
	durs.append(dur[0][1]-dur[0][0])

# print(durs)
median = np.median( durs)
# print(median)
to_keep = np.where(durs <median)[0]


print("Spikes to remove: ",len(list(set(to_keep))))
# try:
spikes_select = np.delete(spikes,to_keep) 
spikes_select2 = spikes[to_keep]
#remove spikes
spikes_wf_sel = np.delete(spikes_wf,to_keep,axis=0) 
spikes_wf_sel2 = spikes_wf[to_keep] 

# except:
	# print("index failed: ",i)


plt.figure(figsize=(20,15))
ax1=plt.subplot(2,1,1)
plt.plot(spikes,np.ones(spikes.shape[0]),'.')
plt.plot(spikes_select,np.ones(spikes_select.shape[0]),'.','r')
plt.subplot(2,1,2, sharex = ax1)
plt.plot(spikes[:-1],isis,'.')
plt.plot(spikes_select2,isis[to_keep],'.')


print(spikes_wf_sel.shape)

#Plot result
plt.figure(figsize=(10,15))
rows = 3
columns= 1


plt.subplot(rows,columns,1)
ax1,ax_fst,ax_last =plot_events(spikes_wf,col='g',tit='spikes',width_ms=50)
set_plot_info([ax_fst,ax_last],["First spike","Last spike"])

plt.subplot(rows,columns,3)
ax1,ax_fst,ax_last =plot_events(spikes_wf_sel2,col='b',tit='spike selection2',width_ms=50)
set_plot_info([ax_fst,ax_last],["First spike","Last spike"])
plt.subplot(rows,columns,2)
ax1,ax_fst,ax_last =plot_events(spikes_wf_sel,col='b',tit='spike selection',width_ms=50)
set_plot_info([ax_fst,ax_last],["First spike","Last spike"])

# plt.savefig(path[:-3]+"png")
if show:
	plt.show()

if save:
	save_files(spikes_select,spikes_wf_sel,path_save,"events_bursts",file_name+"_"+file_ext)
	save_files(spikes_select2,spikes_wf_sel2,path_save,"events_no_bursts",file_name+"_"+file_ext+"2")