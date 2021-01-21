#!/bin/bash

# python3 ../get_spike_events.py -p $1/exp_final.asc -c 0 -e control_final -s False -sv True;
python3 superpos_from_events.py -p1 $1/events/exp_final_control_final_waveform.txt -p2 $1/events/$2 -w 50 -l1 first_control -l2 final_control -c2 g -ti "shape recovery control fst";
python3 superpos_from_events.py -p1 $1/events/exp_final_control_final_waveform.txt -p2 $1/events/$3 -w 50 -l1 last_control -l2 final_control -c2 g -ti "shape recovery control lst";