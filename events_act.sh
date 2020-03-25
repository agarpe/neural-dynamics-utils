#!/bin/bash

# for d in ./../model/slow-fast/*; do
for d in $1*.asc; do
	# echo "$d"
	if [ ! -d "$d" ]; then
		# file_name="$(echo "$dd" | cut -c22-28)"
		# file_name="$(echo "${d##*/}")"
		file_name="$(echo "${d##*/}")"
		file_name="${file_name%%.asc}"
		echo "Analysing $file_name"
		# echo "$file_name"

		python3 "detect_events.py" "$d" 
		# SALE CON .asc!!!
		python3 "burst_charact.py" "$1" "$file_name" "1000"
		# python3 "check_events.py" "$d"
	fi 
done