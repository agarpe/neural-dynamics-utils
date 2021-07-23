#!/bin/bash


# for dir in $1/*/     # list directories in the form "/tmp/dirname/"
# do
dir=$1

dir=${dir%*/}      # remove the trailing "/"
echo ${dir##*/}    # print everything after the final "/"
dir_name=${dir##*/}
line=$(head -n 1 $1/run_info.txt)
# echo $1/$dir_name $line
python3 autom_spike_detection.py ../../data/laser/pyloric/$dir_name $line ;
# python3 autom_spike_detection.py ../../data/laser/single_neuron/$dir_name $line ;
# done