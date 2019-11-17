#!/bin/bash

for d in ./../model/slow-fast/*; do
	# echo "$d"
	if [ ! -d "$d" ]; then
		# file_name="$(echo "$dd" | cut -c22-28)"
		file_name="$(echo "${d##*/}")"
		echo "Analysing $file_name"
		# echo "$file_name"

		python3 "detect_events.py" "$d" 
	fi 
done