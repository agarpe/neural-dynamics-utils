#!/bin/bash
for f in $1*.h5; do 
    echo "$f"
    file="$(echo "${f##*/}")"
    # echo "$aux"
    python3 "plot_hd5.py" "$1" "$file"
done

