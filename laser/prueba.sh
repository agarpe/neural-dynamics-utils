#!/bin/bash

for dir in $1/*/     # list directories in the form "/tmp/dirname/"
do
    dir=${dir%*/}      # remove the trailing "/"
    echo ${dir##*/}    # print everything after the final "/"
    dir_name=${dir##*/}
	python3 stats_plot.py ../../data/laser/single_neuron/$dir_name/events/ | tee ../../data/laser/single_neuron/$dir_name/events/stats.log;
done