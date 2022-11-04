
import math 
import pickle as pkl
import glob
import json
import h5py
import yaml
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from sklearn.preprocessing import minmax_scale
import sys
import argparse 
from scipy.signal import find_peaks
import superpos_functions as laser_utils
import pandas as pd
plt.rcParams.update({'font.size': 13})

dt = 0.001
def get_events(f_data, f_events, ms_r, ms_l, dt=0.001):
	#read data
	data = pd.read_csv(f_data, delimiter = " ",skiprows=2,header=None)
	data = data.values

	#read events
	try:
		events = pd.read_csv(f_events, delimiter = " ",skiprows=2,header=None)
		events = events.values
	except:
		print("Error in file: ",f_events)
		return np.array([])

	points_r = int(ms_r /dt)
	points_l = int(ms_l /dt)

	waveforms = np.empty((events.shape[0],points_l+points_r),float)

	time = data[:,0]

	count =0
	for i,event in enumerate(events[:,0]):
		indx = np.where(time == event)[0][0] #finds spike time reference

		try:
			waveforms[i] =data[indx-points_l:indx+points_r,1]
		except:
			count +=1
			# print(i)

	print(count, "events ignored")
	# print(waveforms)
	return waveforms[2:-2] #Ignore 2 first events, usally artefacts

def get_waveforms(f,f_events,ms_l=100, ms_r=100):
	waveforms = []

	waveforms_file = f[:-4] + '_waveforms.pkl'

	try:
		print("Trying to load",waveforms_file)
		with open(waveforms_file,'rb') as f_waveforms:
			waveforms = pkl.load(f_waveforms)

	except Exception as e:
		print("Error loading", e)
		print("Calculating waveforms")

		waveforms = get_events(f,f_events,ms_r, ms_l)
		print(waveforms.shape)
		
		print("Writing",waveforms_file)

		with open(waveforms_file,'wb') as f_waveforms:
			pkl.dump(waveforms, f_waveforms)

	return waveforms

def plot_df_by(qs_df, temps, ref_label):
	nrows = 4
	ncols = math.ceil(len(qs_df.columns)/nrows)
	fig, axes = plt.subplots(ncols=ncols,nrows=nrows,figsize=(ncols*4,nrows*3))
	axes = axes.flat
	norm_temps = minmax_scale(np.array(temps))
	# print(norm_temps)
	for i,column in enumerate(qs_df):
		# print(qs_df[column].index)
		# print("\n\n\n")
		# print(qs_df.loc[temps, column])
		axes[i].scatter(temps, qs_df[column], 
						s=1/norm_temps*plt.rcParams['lines.markersize'],color=colors)
		axes[i].set_xlabel(ref_label)
		axes[i].set_ylabel(column)

		# for x, y in zip(temps, qs_df.loc[temps, column]):
		#     axes[i].text(x, y, str(y), color="red", fontsize=12)
		# axes[i].margins(0.1)

def q10_formula(const, q10, dT):
	return const * (q10 ** (dT/10)) if q10 > 0 else 1*const

def get_q10_value(params, key_ref='Q10_'):
	static_items = dict(params).items()

	# print("name original q10 new")
	for key, value in static_items:   # iter on both keys and values
		if key.startswith(key_ref):
			param = key[len(key_ref):]
			if param not in params.keys():
				params[param]=1
			qs[param] = q10_formula(params[param], params[key], params['diff_T'])
			if param == 'Q10_Gd':
				print(param, qs[param])
			# print(params['diff_T'], key, params[param], params[key], qs[param])
	return qs

def get_logs(params_path):	
	with open(params_path) as log:
	    data = log.read()
	return json.loads(data)

def save_as_yaml(dict_file, path):
	print("Saving parameters at ",path)
	with open(path, 'w') as file:
		documents = yaml.dump(dict_file, file)


# colors = ['teal', 'lightsalmon', 'darkseagreen','maroon','teal', 'brown', 'blue', 'green','maroon']
colors=['b', 'r', 'g', 'brown', 'teal', 'maroon', 'lightsalmon', 'darkseagreen', 'k']

ap = argparse.ArgumentParser()

ap.add_argument("-p", "--path", required=True, help="Path to the file to analyze")
ap.add_argument("-c","--cols",required=False, default=None,help="Index to the elements in the trial separated by space")
ap.add_argument("-t","--trials",required=False, default=None,help="Selection of trials to plot")
ap.add_argument("-nn", "--neu_name", required=False, default='', help="Extension of the events file p.e. for '*_v1_spikes.asc' -nn _v1")

args = vars(ap.parse_args())

path = args['path']
param_name = args['neu_name']

files = sorted(glob.glob(path+"*[!_spikes].asc"))

if len(files) == 0:
	print("No files to plot")
	exit()

# files = files[:5]
# print(files)

# plt.gca().set_color_cycle(colors)

# durations = {}
all_qs = {}
all_waveforms = []
all_labels = []
for i,f in enumerate(files):

	qs = {}
	params_path = f[:-4] + '_params.log'
	f_events = f[:-4]+param_name+"_spikes.asc"
	file_name = f[f.rfind("/")+1:]

	print(f)
	print(f_events)
	print(params_path)

	params = get_logs(params_path)

	save_as_yaml(params, params_path[:-4]+'.yaml')

	waveforms = get_waveforms(f,f_events,ms_l=40, ms_r=40)
	if len(waveforms) == 0:
		print("Skiping",f)
		continue

	#align to the peak
	waveforms = np.array([w-np.max(w) for w in waveforms])
	a_waveform = waveforms[len(waveforms)//2]
	all_waveforms.append(a_waveform )
	all_labels.append(params['diff_T'])

	# for spike in waveforms:
	# All spikes the same in the model, get middle one
	dur_refs,th = laser_utils.get_spike_duration(a_waveform, 0.001)
	# plt.plot(a_waveform)
	# plt.show()
	duration = dur_refs[1]-dur_refs[0]
	print("Duration value:", duration)


	amplitude = laser_utils.get_spike_amplitude(a_waveform, 0.001)
	print("Amplitude value:", amplitude)

	# durations[params['diff_T']] = duration

	qs['amplitude'] = amplitude
	qs['duration'] = duration
	try:
		qs['Q10'] = params['general_Q10']
	except:
		qs['Q10'] = params['Q10_f']

	qs['diff_T'] = params['diff_T']

	qs = get_q10_value(params)
	qs['cm'] = params['cm'] + params['cm'] * params['gamma_T'] * params['diff_T']

	# all_qs[params['diff_T']] = qs
	all_qs[file_name] = qs

qs_df = pd.DataFrame.from_dict(all_qs, orient='index')
print(qs_df)


colors = plt.cm.coolwarm(np.linspace(0,1,qs_df['diff_T'].size))
# colors = plt.cm.get_cmap('Oranges')
from cycler import cycler
plt.gca().set_prop_cycle(cycler('color', colors))


all_waveforms = np.array(all_waveforms)
time = np.arange(all_waveforms.shape[1])*dt

title = path[path[:-1].rfind('/')+1:-1]

# tam = (15,10)
plt.figure()
plt.gca().set_prop_cycle(cycler('color', colors))
plt.plot(time, all_waveforms.T)
plt.title(title)
plt.legend(all_labels)
plt.tight_layout()
plt.savefig(path+"shape.png",dpi=200)

small_waveforms = all_waveforms[:,int(90/dt):-int(60/dt)]
time = np.arange(small_waveforms.shape[1])*dt

plt.figure()
plt.gca().set_prop_cycle(cycler('color', colors))
plt.plot(time, small_waveforms.T)
plt.title(title)
plt.tight_layout()
plt.legend(all_labels)
plt.savefig(path+"shape_zoom.png",dpi=200)



qs_df.plot.scatter('diff_T','duration',color=colors)
plt.xlabel(u"ΔT")
plt.ylabel("Spike duration (ms)")
plt.title(title)
# plt.show()
plt.savefig(path+"durations_dt.png",dpi=200)

qs_df.plot.scatter('Q10','duration',color=colors)
plt.xlabel(u"ΔQ10")
plt.ylabel("Spike duration (ms)")
plt.title(title)
# plt.show()
plt.savefig(path+"durations_dq10.png",dpi=200)

qs_df.plot.scatter('diff_T','amplitude',color=colors)
plt.xlabel(u"ΔT")
plt.ylabel("Spike amplitude")
plt.title(title)
# plt.show()
plt.savefig(path+"amplitude_dt.png",dpi=200)

temps = qs_df['diff_T'].values
# norm_temps = minmax_scale(np.array(temps))

plot_df_by(qs_df, temps,u"ΔT")
plt.suptitle(title)

plt.tight_layout()
plt.savefig(path+"parameters_dt.png",dpi=200)

temps = qs_df['Q10'].values

plot_df_by(qs_df, temps,u"ΔQ10")

plt.suptitle(title)
plt.tight_layout()
plt.savefig(path+"parameters_Q10.png",dpi=200)

# plt.show()