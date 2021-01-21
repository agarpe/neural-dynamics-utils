import os,sys

if len(sys.argv)==6:
	e_min = int(sys.argv[4])
	e_max = int(sys.argv[5])
	wind = int(sys.argv[3])
	sufix = sys.argv[2]
	path = sys.argv[1]

else:
	print("Use: python3 autom_superpos.py path sufix window(ms) min_exp max_exp\n")
	print("Example: python3 autom_superpos.py ../../data/laser/single_neuron/27-Jul-2020 5400_50f 50 1 5")
	exit()

path += "/events"

#Plot all experiments grid pre-laser-pos
for i in range(e_min,e_max+1):
	print("Exp ",i)
	os.system("python3 superpos_from_events_3.py %s/exp%d_%s %d"%(path,i,sufix,wind))

# ------------------------------------------------

#Plot single case of first-last control
#Labels for Control-Control

label1 = "First control. N. spikes: "
label2 = "Last control. N. spikes: "
color2 = 'g'
title = "First Last control"

print("first-last control")
os.system("python3 superpos_from_events.py -p1 %s/exp%d_%s_control_pre_waveform.txt -p2 %s/exp%d_%s_control_pre_waveform.txt -w %d -l1 '%s' -l2 '%s' -c2 '%s' -t '%s' -sh 'n'"%(path,e_min,sufix,path,e_max,sufix,wind,label1,label2,color2,title))

#------------------------------------------------

#Plot single case of first-last laser
# Labels for Laser-Laser
label1 = "First laser. N. spikes: "
label2 = "Last laser. N. spikes: "
color2 = 'r'
title = "First Last laser"

print("first-last laser")
os.system("python3 superpos_from_events.py -p1 %s/exp%d_%s_laser_waveform.txt -p2 %s/exp%d_%s_laser_waveform.txt -w %d -l1 '%s' -l2 '%s' -c2 '%s' -t '%s' -sh 'n'"%(path,e_min,sufix,path,e_max,sufix,wind,label1,label2,color2,title))

#------------------------------------------------

#Analyze general plot files 
os.system("python3 stats_plot.py %s/ > 'amplitudes_log.log'"%(path)) 
