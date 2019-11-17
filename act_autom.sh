#!/bin/bash

for d in ./../model/slow-fast/*; do	
	# echo "$d"
	folder="$(echo "${d##*/}")"
	# echo "$folder"
	if [ -d "$d" ] && [ "$folder" != "dataview" ]; then
        for dd in "$d"/*; do
        	if [ "$dd" != ".." ] && [ "$dd" != "." ]; then
        		file_name="$(echo "${dd##*/}")"
    #     		echo "$file_name"
				# echo "${dd##*/}"
				echo "$folder"
				aux="$(echo "${folder##*_syns_*}")"
				# echo "$var"
				if [ "$aux" != "" ]; then #not syns
					# echo "not syns"
					python3 "burst_charact.py" "$folder" > "results_invariant_tests/stats/slow-fast_$folder.txt"
				fi
        		break
        	fi
        done
    fi
done