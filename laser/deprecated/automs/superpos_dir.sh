#!/bin/bash


dir=$1

dir=${dir%*/}      # remove the trailing "/"
echo ${dir##*/}    # print everything after the final "/"
dir_name=${dir##*/}
line=$(head -n 1 $1/run_info.txt)
# echo $1/$dir_name $line
python3 autom_superpos.py ../../data/laser/single_neuron/$dir_name $line ;
