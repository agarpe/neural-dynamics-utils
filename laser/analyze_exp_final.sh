#!/bin/bash

# python3 ../get_spike_events.py -p $1/exp_final.asc -c 0 -e control_final -s False -sv True;
python3 superpos_from_events.py $1/events/exp_final_control_final_waveform.txt $1/events/$2 50 first_control final_control g "shape recovery control fst";
python3 superpos_from_events.py $1/events/exp_final_control_final_waveform.txt $1/events/$3 50 last_control final_control g "shape recovery control lst";