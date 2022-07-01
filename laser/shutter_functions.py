import pandas as pd
import numpy as np

import superpos_functions as sf

def get_durs(waveforms):

	durations =[]
	for i,spike in enumerate(waveforms):
		spike = sf.align_spike(spike, 40, 0.1,i )

		if len(spike) ==0:
			durations.append(0)
			continue

		vals,th = sf.get_spike_duration(spike,dt=0.1,tol=1)
		# plt.plot(spike)
		# vals = np.array(vals)
		# plt.plot(vals/0.1,(th,th),'x')
		# plt.title(str(vals[1]-vals[0]))
		# plt.show()
		try:
			durations.append(vals[1]-vals[0])
		except:
			print(vals)
			continue

	return np.array(durations)

def read_data(path, ctype):
	waveforms =  sf.read_from_events(path+'_%s_waveform_single.txt'%ctype,max_cols=80/0.1,dt=0.1)

	waveforms = waveforms.values
	print(waveforms.shape)

	try:
		stim_path = path + '_%s_shutter_time_references.txt'%ctype # name from v1
		# stim_path = path[:path.rfind('/')] + 'laser_shutter_time_references.txt'
		print(stim_path)
		stim = np.loadtxt(stim_path)
		print("Stim ",stim.shape)
	except Exception as e:
		stim = []
		if ctype == 'laser':
			print("EXCEPTION:",e.args)
		pass

	return waveforms, stim



def get_durs_from_file(path, ctype):

	waveforms, stim = read_data(path, ctype)
	durations = get_durs(waveforms)

	try:
		stim = stim[np.where(durations>2)]
	except:
		pass

	durations = durations[np.where(durations>2)]

	return durations, stim