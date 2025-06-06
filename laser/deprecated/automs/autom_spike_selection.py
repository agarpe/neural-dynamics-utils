import os,sys



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
	# python3 spike_selection.py -p pruebas/ -ps pruebas/ -fe sel -sh 'n' -sa 'y' -fn exp3-oct23_control_pre
	os.system("python3 spike_selection.py -p %sevents/ -fn exp%d_%s_control_pre -ps %s -fe sel -sh 'n' -sa 'y'"%(path,i,sufix,path))
	os.system("python3 spike_selection.py -p %sevents/ -fn exp%d_%s_laser -ps %s -fe sel -sh 'n' -sa 'y'"%(path,i,sufix,path))
	os.system("python3 spike_selection.py -p %sevents/ -fn exp%d_%s_control_pos -ps %s -fe sel -sh 'n' -sa 'y'"%(path,i,sufix,path))

