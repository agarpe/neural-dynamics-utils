#Calls autom_spike_detection.py and autom_superpos.py
#Detects spikes and then generate superpos for control_pre-laser-control_pos

import os,sys


if len(sys.argv)==6:
	e_min = int(sys.argv[4])
	e_max = int(sys.argv[5])
	width = int(sys.argv[3])
	sufix = sys.argv[2]
	path = sys.argv[1]

else:
	print("Use: python3 autom_spike_n_superpos.py path sufix window(ms) min_exp max_exp\n")
	print("Example: python3 autom_spike_n_superpos.py ../../data/laser/single_neuron/27-Jul-2020 5400_50f 50 1 5")
	exit()

cmd_spike ="python3 ./autom_spike_detection.py %s %s %d %d %d "%(path,sufix,width,e_min,e_max)
cmd_superpos ="python3 ./autom_superpos.py %s %s %d %d %d "%(path,sufix,width,e_min,e_max) 
os.system(cmd_spike)
os.system(cmd_superpos)


