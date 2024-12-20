# automatically get .asc files from all trials with same sufix from min_exp to max_exp
# and get spikes events for control_pre laser and control_pos
# control_pre -> column 0
# laser -> column 1
# control_pos -> column 2

import os,sys

general_path="~/Workspace/scripts"

if len(sys.argv)==6:
	e_min = int(sys.argv[4])
	e_max = int(sys.argv[5])
	wind = int(sys.argv[3])
	sufix = sys.argv[2]
	path = sys.argv[1]

else:
	print("Use: python3 autom_spike_detection.py path sufix window(ms) min_exp max_exp\n")
	print("Example: python3 autom_spike_detection.py ../../data/laser/single_neuron/27-Jul-2020 5400_50f 50 1 5")
	exit()

print("Detecting spikes...")

for i in range(e_min,e_max+1):
	os.system("python3 %s/get_spike_events.py -p %s/exp%d_%s.asc -c 0 -e \"control_pre\" -sh n -sv y"%(general_path,path,i,sufix))
	os.system("python3 %s/get_spike_events.py -p %s/exp%d_%s.asc -c 1 -e \"laser\" -sh n -sv y"%(general_path,path,i,sufix))
	os.system("python3 %s/get_spike_events.py -p %s/exp%d_%s.asc -c 2 -e \"control_pos\" -sh n -sv y"%(general_path,path,i,sufix))

