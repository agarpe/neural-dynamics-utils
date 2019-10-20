#!/bin/bash

for d in ./../model/slow-fast/*; do
	if [ -d "$d" ]; then
		echo "$d"
        for dd in "$d"/*; do
        	if [ "$dd" != ".." ] && [ "$dd" != "." ] && [ "$dd" != "dataview" ]; then
        		file_name="$(echo "$dd" | cut -c22-35)"
        		echo "$file_name"
        		python3 "burst_charact.py" "$file_name" > "results_invariant_tests/stats/slow-fast_$file_name.txt"
        		break
        	fi
        done
    fi
done