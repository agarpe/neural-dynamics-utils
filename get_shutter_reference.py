import sys
sys.path.append('~/Workspace/scripts/')
# sys.path.append('..')

import charact_utils as cu

import pandas as pd
import argparse

import numpy as np
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("-pspk", "--path_spike", required=True, help="Path to spike times file")
ap.add_argument("-pshu", "--path_shutter", required=True, help="Path to shutter data file")
ap.add_argument("-onoff", "--onoff", required=True, help="Spike events as on-off")
ap.add_argument("-sn", "--save_name", required=True, help="Path to generated waveform files")
ap.add_argument("-dt","--dt",required=True,help="Sampling period in ms")
ap.add_argument("-c","--column",required=True,help="Column for signal data")
ap.add_argument("-err", "--error", required=False, default=100, help="Time allowed to initiate shutter after spike. 100ms default")
ap.add_argument("-verb", "--verbose", required=False, default='n', help="Verbose process y/n")
ap.add_argument("-sa", "--save", required=False, default='y', help="Option to save plot file")
ap.add_argument("-sh", "--show", required=False, default='n', help="Option to show plot file")


args = vars(ap.parse_args())

spikes_file_name = args["path_spike"]
events_file_name = args["path_shutter"]
savepath = args["save_name"]

verbose = True if args['verbose']=='y' else False 

show= True if args['show']=='y' else False 
save= True if args['save']=='y' else False 
# ws_l = int(args['window_width_l'])
# ws_r = int(args['window_width_r'])

# width_ms = int(args["window_width"])

dt = float(args['dt'])
col = int(args['column'])
onoff=True if args['onoff']=='y' else False 
error = float(args['error'])


# try:
# 	signal = pd.read_csv(sig_file_name,delimiter=' ',header=None)[col]
# except ValueError:
# signal = pd.read_csv(sig_file_name,delimiter=' ',header=None,skiprows=1)[col]

try:
	events = pd.read_csv(events_file_name,delimiter=' ',header=None,skiprows=1)[0].values
except ValueError:
	events = pd.read_csv(events_file_name,delimiter=' ',header=None,skiprows=2)[0]

# try:
# except ValueError:
spikes = cu.read_spike_events(spikes_file_name,onoff=onoff,skiprows=0,col=0,dataview=False)

# signal *= scale

print("signal shape",spikes.shape)
print("events shape",events.shape)
print(events)

time = np.arange(0,events.shape[0],1.0)
time *= dt
# plt.plot(spikes, np.ones(spikes.shape),'|')
# plt.plot(time, events)
# plt.show()


onoff_events = cu.get_onoff(events)

offs = onoff_events[:,1]
ons = onoff_events[:,0]
# plt.plot(onoff_events,np.ones(onoff_events.shape),'.')
# plt.plot(time, events)
# plt.show()

# error = 100 # error in ms
spk_id = 0
shutter_id = 0

#reference as [distance ini, distance end] to shutter event
refs = np.empty((spikes.shape[0], 2))
refs[:] = np.NaN

for i,spike in enumerate(spikes):
	try:
		on = ons[shutter_id]
		off = offs[shutter_id]
	except IndexError:
		break
	except Exception as e:
		print("EXCEPTION: ",e.args)


	if(on > (spike + error)): # activation after spike --> skip
		if verbose:
			print(i, "Skiping activation after spike")
		refs[i,0] = np.nan
		refs[i,1] = np.nan
		continue


	while(shutter_id+1 < len(ons) and 
		ons[shutter_id+1] < (spike + error)): # two shutters in 1 ISI
		if verbose:
			print(i, shutter_id, "Two shutters in 1 ISI")
			print("\t", ons[shutter_id+1], spike+error)
		shutter_id+=1


	if(i==0 or spikes[i-1] < off and off < (spike + error)):
		if verbose:
			print(i, shutter_id, "All ok")
		try:
			refs[i, 0] = spike - ons[shutter_id]
			refs[i, 1] = spike - offs[shutter_id]
		except IndexError:
			pass
		except Exception as e:
			print("EXCEPTION: ",e.args)
		shutter_id += 1
	else:
		if verbose:
			print("Error ", spikes[i-1], " !< ", off," !< ", spike)

# print(refs)


plt.plot(spikes, np.ones(spikes.shape),'|')
plt.plot(onoff_events,np.ones(onoff_events.shape),'.')
# plt.plot(time, events)

for i,d in enumerate(refs):
	plt.annotate('id%d\nini%.2f\nend%.2f'%(i,d[0],d[1]),
            xy=(spikes[i], 1.1), xycoords='data',
            # horizontalalignment='left', verticalalignment='top',
            fontsize=10)

print(onoff_events.shape)
for i,d in enumerate(onoff_events):
	plt.annotate('id%d'%(i),
            xy=(d[0], 1.05), xycoords='data',
            # horizontalalignment='left', verticalalignment='top',
            fontsize=10)

plt.ylim(0,2.5)
if show:
	plt.show()

if save:
	print("Writing waveforms at",savepath)
	print(np.array(refs).shape)
	np.savetxt(savepath,refs)