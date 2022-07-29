import os
import pandas as pd
import argparse
import numpy as np
import matplotlib.pyplot as plt
import superpos_functions as sf
from shutter_functions import *

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=True, help="Path to the file to show stats from")
ap.add_argument("-p2", "--path2", required=False, help="Path 2 to the file to show stats from")
ap.add_argument("-se", "--save_extension", required=False, default='', help="Extension to path where file is saved")
ap.add_argument("-sa", "--save", required=False, default='n', help="Option to save plot file")
ap.add_argument("-sh", "--show", required=False, default='y', help="Option to show plot file")
# ap.add_argument("-v", "--verbrose", required=False, default='n', help="Option to verbrose actions")
# ap.add_argument("-log", "--log", required=False, default='n', help="Option to log best trial selection")
args = vars(ap.parse_args())


path = args['path']
path2 = args['path2']
# plot_mode = args['mode'] 
# data_type = args['data_type']

# ext_path = args['path_extension'] #subpath where 
save_extension = args['save_extension'] #subpath where 
# sort_by = args['sort_by']
show= True if args['show']=='y' else False 
save= True if args['save']=='y' else False 
# log= True if args['log']=='y' else False 
# verb= True if args['verbrose']=='y' else False 
# spike_selection= True if args['selection']=='y' else False 
# cols=int(args['cols'])


waveforms, stim = read_data(path, 'laser')
durations = get_durs(waveforms)

stim = stim[np.where(durations>2)]
durations = durations[np.where(durations>2)]

# stim = stim*-1
# print(np.where(stim[:,1]>650)[0])

# plt.plot(waveforms[np.where(stim[:,1]>650)].T)
# plt.show()



plt.figure()
plt.scatter(stim[:,1],durations, s=10, label='laser')
# plt.vlines(stim[:,1],9,durations, label='laser')
plt.scatter(np.nanmedian(stim[:,1]), np.mean(durations), s=50,label='laser_median')
plt.xlabel("Time end shutter - spike (ms)")
plt.ylabel("Spike duration (ms)")
plt.title(path)



path_images = path[:path.rfind("/")] + '/images/shutter/' + path[path.rfind('/'):] 

os.system("mkdir -p %s"%path_images[:path_images.rfind('/')])
savepath = path_images + "_shutter_comparation"
print("Saving fig at",savepath)
plt.savefig(savepath + '.png')
plt.savefig(savepath + '.pdf',format='pdf')


#Plot with controls

waveforms_control,a = read_data(path, 'control')
waveforms_recovery,a = read_data(path, 'recovery')

durations_control = get_durs(waveforms_control)
durations_recovery = get_durs(waveforms_recovery)

durations_control= durations_control[np.where(durations_control > 9)]
durations_recovery= durations_recovery[np.where(durations_recovery > 9)]


plt.figure()
plt.scatter(stim[:,1],durations, s=10, label='laser')
# plt.vlines(stim[:,1],9,durations, label='laser')
plt.scatter(np.nanmedian(stim[:,1]), np.mean(durations), s=50,label='laser_median')
plt.scatter(np.ones(durations_control.shape),durations_control, s=10,label='control')
plt.scatter(np.ones(durations_recovery.shape)*10,durations_recovery, s=10,label='recovery')
plt.scatter(1, np.median(durations_control), s=50,label='control-median')
plt.scatter(10, np.median(durations_recovery), s=50,label='recovery-median')

plt.legend()
plt.xlabel("Time from end of shutter to spike (ms)")
plt.ylabel("Spike duration (ms)")
plt.title(path)


savepath = path_images+"_controls_shutter_comparation"
print("Saving fig at",savepath)
plt.savefig(savepath + '.png')
plt.savefig(savepath + '.pdf',format='pdf')


if show:
	plt.show()




# waveforms2, stim2 = read_data(path2, 'laser')
# durations2 = get_durs(waveforms2)

# stim2 = stim2[np.where(durations2>2)]
# durations2 = durations2[np.where(durations2>2)]

# plt.figure()
# plt.scatter(stim[:,1],durations, s=10, label='laser 1')
# plt.scatter(stim2[:,1],durations2, s=10, label='laser 2')
# # plt.vlines(stim[:,1],9,durations, label='laser')
# # plt.scatter(np.nanmedian(stim[:,1]), np.mean(durations), s=50,label='laser_median')
# plt.xlabel("Time end shutter - spike (ms)")
# plt.ylabel("Spike duration (ms)")
# plt.legend()
# plt.title(path)

# plt.show()