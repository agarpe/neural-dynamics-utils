import charact_utils as utils 
import matplotlib.pyplot as plt
import argparse

def read_n_plot(path,title,limit=-1):
	spikes = utils.read_spike_events(path)
	isis = utils.get_ISI(spikes)
	if limit >0:
		isis_y = isis[1:limit+1]
	else:
		isis_y = isis[1:]
	print(limit)
	plt.plot(isis[:limit],isis_y,'.')
	plt.xlabel("ISI(t)")
	plt.ylabel("ISI(t+1)")
	plt.title(title)


ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=True, help="Path to the file with the spike events")
ap.add_argument("-sh", "--show", required=False,default='y', help="Show plot")
ap.add_argument("-sv", "--save", required=False,default='y', help="Save events")

args = vars(ap.parse_args())
path = args['path']
path_control_pre = path+"_control_pre_events.txt"
path_laser = path+"_laser_events.txt"
path_control_pos = path+"_control_pos_events.txt"


show= True if args['show']=='y' else False 
save= True if args['save']=='y' else False 


plt.figure(figsize=(5,30))
plt.subplot(3,2,1)
read_n_plot(path_control_pre,"control_pre",)
plt.subplot(3,2,2)
read_n_plot(path_control_pre,"control_pre",limit=1000)
plt.subplot(3,2,3)
read_n_plot(path_laser,"laser")
plt.subplot(3,2,4)
read_n_plot(path_laser,"laser",limit=1000)
plt.subplot(3,2,5)
read_n_plot(path_control_pos,"control_pos")
plt.subplot(3,2,6)
read_n_plot(path_control_pos,"control_pos",limit=1000)

plt.tight_layout()
if show:
	plt.show()

# spikes = utils.read_spike_events(path)

# plt.plot(spikes)
# plt.show()

# isis = utils.get_ISI(spikes)

# plt.plot(isis[:-1],isis[1:],'.')
# plt.show()