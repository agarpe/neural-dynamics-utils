import sys
sys.path.append('~/Workspace/scripts/')
import charact_utils as cu

import pandas as pd
import argparse
import numpy as np


ap = argparse.ArgumentParser()
# ap.add_argument("-ps", "--path_signal", required=True, help="Path to signal data file")
ap.add_argument("-pe", "--path_events", required=True, help="Path to events data file")
# ap.add_argument("-w", "--window_width", required=True, help="Half window width in ms")
ap.add_argument("-onoff", "--onoff", required=True, help="Events as on-off")
ap.add_argument("-sn", "--save_name", required=True, help="Path to generated waveform files")
ap.add_argument("-dt","--dt", required=True, help="Sampling period in ms")
ap.add_argument("-c","--column", required=True, help="Column for signal data")
ap.add_argument("-th","--threshold", required=False, help="Threshold onoff")

args = vars(ap.parse_args())

# sig_file_name = args["path_signal"]
events_file_name = args["path_events"]
savepath = args["save_name"]
# width_ms = int(args["window_width"])
dt = float(args['dt'])
th = float(args['threshold'])
col = int(args['column'])
onoff=True if args['onoff']=='y' else False 


# try:
	# data = pd.read_csv(events_file_name,delimiter=' ',header=None)[col]
# except ValueError:
data = pd.read_csv(events_file_name,delimiter=' ',header=None,skiprows=2)[col]

# try:
# 	data = cu.read_spike_events(events_file_name,onoff=onoff,col=col,dataview=False)
# except ValueError:
# 	data = cu.read_spike_events(events_file_name,onoff=onoff,skiprows=2,col=col,dataview=False)

print(data.shape)
events = cu.get_onoff(data, dt, th)

print(events.shape)

f = open(savepath,'w')
np.savetxt(f,events)

f.close()


# print("signal shape",signal.shape)
# print("events shape",events.shape)
# print("Writing waveforms at",savepath)
# waveform = cu.save_waveforms_new(signal,events,savepath,width_ms,dt=dt,split=False,onoff=False)