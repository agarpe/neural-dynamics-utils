import sys
sys.path.append('~/Workspace/scripts/')
import charact_utils as cu

import pandas as pd
import argparse

import numpy as np
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("-ps", "--path_signal", required=True, help="Path to signal data file")
ap.add_argument("-pe", "--path_events", required=True, help="Path to events data file")
ap.add_argument("-wl", "--window_width_l", required=False, default=0, help="Half window width left in ms")
ap.add_argument("-wr", "--window_width_r", required=False, default=0, help="Half window width left in ms")
ap.add_argument("-onoff", "--onoff", required=True, help="Events as on-off")
ap.add_argument("-sn", "--save_name", required=True, help="Path to generated waveform files")
ap.add_argument("-dt","--dt",required=True,help="Sampling period in ms")
ap.add_argument("-c","--column",required=True,help="Column for signal data")
ap.add_argument("-osc","--output_scale",required=False, default=1, help="Output scale")

args = vars(ap.parse_args())

sig_file_name = args["path_signal"]
events_file_name = args["path_events"]
savepath = args["save_name"]


ws_l = int(args['window_width_l'])
ws_r = int(args['window_width_r'])

# width_ms = int(args["window_width"])

dt = float(args['dt'])
col = int(args['column'])
scale = float(args['output_scale'])
onoff=True if args['onoff']=='y' else False 


# try:
# 	signal = pd.read_csv(sig_file_name,delimiter=' ',header=None)[col]
# except ValueError:
signal = pd.read_csv(sig_file_name,delimiter=' ',header=None,skiprows=1)[col]

try:
	events = pd.read_csv(events_file_name,delimiter=' ',header=None)
except ValueError:
	events = pd.read_csv(events_file_name,delimiter=' ',header=None,skiprows=2)

# try:
# except ValueError:
# 	events = cu.read_spike_events(events_file_name,onoff=onoff,skiprows=2,col=0,dataview=False)

signal *= scale

print("signal shape",signal.shape)
print("events shape",events.shape)
print("Writing waveforms at",savepath)
# waveform = cu.save_waveforms_new(signal,events,savepath,width_ms,dt=dt,split=False,onoff=False)
# waveform = cu.save_waveforms_new(signal,events,savepath,width_ms_l, width_ms_r,dt=dt,split=False,onoff=False) #onoff False because they have been read as single events.
waveforms = cu.save_waveforms_from_signal(signal,events,savepath,dt=dt,width_l=ws_l, width_r=ws_r) 

time = np.arange(0,signal.shape[0],1.0)
time *= dt

#save events references from max

max_id = np.argmax(waveforms)

# ini --> difference from max to ini == argmax
# end --> difference from max to end 
refs = [(np.argmax(waveform)*dt + ws_l, (waveform.size - np.argmax(waveform))*dt + ws_r) for waveform in waveforms.T]
# ini_ref = max_id
# end_ref = waveform.size - max_id

savepath = savepath[:-4]+"_references.txt"
print("Writing waveforms references at",savepath)

print(np.array(refs).shape)
np.savetxt(savepath,refs)



# plt.plot(time,signal)
# plt.plot(events,np.zeros(events.shape),'.')
# plt.show()

# plt.plot(waveforms.T)
# plt.plot(refs,np.ones((len(refs),2)),'.','b')
# plt.show()