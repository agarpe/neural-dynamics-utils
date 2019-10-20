#!/bin/bash

for d in ./../model/slow-fast/*; do
	echo "$d"
	if [ ! -d "$d" ]; then
		echo "Analysing $d"
		file_name="$(echo "$dd" | cut -c22-28)"
		python3 "detect_events.py" "$d" 
	fi 
done