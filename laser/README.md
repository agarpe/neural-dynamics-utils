# Neuron_charact

Different neuron characterization scripts.

## Spike superposition
### Files:
```bash
superpos_from_events.py
superpos_from_events_3.py
```

### Description:
Plot the superposed spikes events in different conditions. 
"Normalizes" the event starting and ending point and centralices spike by spike.

### Use:
```bash
python3 superpos_from_events_3.py path/exp_name centered_range
python3 superpos_from_events.py path/exp_name centered_range
```

Loads corresponding events files to path/exp_name adding the sufixes:
	exp_name_control_pre_events.txt
	exp_name_control_laser_events.txt
	exp_name_control_pos_events.txt (only for superpos_from_events_3)

Example:
	python3 superpos_from_events_3.py ../data/laser/15-Jul-2020/exp4_60mW_50f 10
	python3 superpos_from_events_2.py ../data/laser/15-Jul-2020/exp4_60mW_50f 10

### Pre-processing data
This script receives the events from dataview where each row in the file contains all the voltage values corresponding to one same event. 

For the prepocessing the following steps in Dataview are needed:

1. Detect the spikes using Event edit--Template recognition.
	1. Set two vertcursos (Curors--multivert add)
	2. Select trace and capture (choose the right channel)
	3. check optimally scaled/offset
	4. Press analize
	**If there was a good result press Save to save the configuration parameters**

2. Export the event's waveform using:
	1. Event analyse--Save events waveforms
	2. Choose channel
	3. Check Spreadsheet option (this will save only the correspondign trace, saving each event in a row).
	**Save the events with the corresponding sufix: (control_pre_events, laser_events, control_pos_events**



