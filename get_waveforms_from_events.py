import sys
sys.path.append('~/Workspace/scripts/')
import charact_utils as cu

import pandas as pd
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-ps", "--path_signal", required=True, help="Path to signal data file")
ap.add_argument("-pe", "--path_events", required=True, help="Path to events data file")
ap.add_argument("-w", "--window_width", required=False, default=None, help="Half window width in ms")
ap.add_argument("-wl", "--window_width_l", required=False, default=None, help="Half window width left in ms")
ap.add_argument("-wr", "--window_width_r", required=False, default=None, help="Half window width left in ms")
ap.add_argument("-onoff", "--onoff", required=True, help="Events as on-off")
ap.add_argument("-sn", "--save_name", required=True, help="Path to generated waveform files")
ap.add_argument("-dt","--dt",required=True,help="Sampling period in ms")
ap.add_argument("-c","--column",required=True,help="Column for signal data")

args = vars(ap.parse_args())

sig_file_name = args["path_signal"]
events_file_name = args["path_events"]
savepath = args["save_name"]


if(args['window_width'] is not None):
	width_ms_l = int(args['window_width'])
	width_ms_r = int(args['window_width'])

	if args['window_width_l'] is not None and args['window_width_r'] is not None:
		print("Warning: ignoring window_width_r a/o window_width_l values")

else:
	width_ms_l = int(args['window_width_l'])
	width_ms_r = int(args['window_width_r'])

# width_ms = int(args["window_width"])

dt = float(args['dt'])
col = int(args['column'])
onoff=True if args['onoff']=='y' else False 


try:
	signal = pd.read_csv(sig_file_name,delimiter=' ',header=None)[col]
except ValueError:
	signal = pd.read_csv(sig_file_name,delimiter=' ',header=None,skiprows=2)[col]

try:
	events = cu.read_spike_events(events_file_name,onoff=onoff,col=0,dataview=False)
except ValueError:
	events = cu.read_spike_events(events_file_name,onoff=onoff,skiprows=2,col=0,dataview=False)


print("signal shape",signal.shape)
print("events shape",events.shape)
print("Writing waveforms at",savepath)
# waveform = cu.save_waveforms_new(signal,events,savepath,width_ms,dt=dt,split=False,onoff=False)
waveform = cu.save_waveforms_new(signal,events,savepath,width_ms_l, width_ms_r,dt=dt,split=False,onoff=False) #onoff False because they have been read as single events.