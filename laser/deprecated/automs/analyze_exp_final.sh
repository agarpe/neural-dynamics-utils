#!/bin/bash

# Get superpos image from the "final experiment" aka. the final long control after laser trials; compared to 
# $2 and $3 = [].txt waveform files

general_path=~/Workspace/scripts

if [ "$#" -lt "3" ]; then
	echo "\$1 = path \n \$2 = file1 \n \$ = file2 "	
else
	path=$1
	f1=$2
	f2=$3
	# python3 $general_path/get_spike_events.py -p $path/exp_final.asc -c 0 -e control_final -s False -sv True;
	python3 $general_path/laser/superpos_from_events.py -p1 $path/events/exp_final_control_final_waveform.txt -p2 $path/events/$f1 -w 50 -l1 first_control -l2 final_control -c2 g -ti "shape recovery control fst";
	python3 $general_path/laser/superpos_from_events.py -p1 $path/events/exp_final_control_final_waveform.txt -p2 $path/events/$f2 -w 50 -l1 last_control -l2 final_control -c2 g -ti "shape recovery control fst";
fi

