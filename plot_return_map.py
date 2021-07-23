import charact_utils as utils 
import matplotlib.pyplot as plt
import argparse
#TODO: split in two functions
def read_n_plot(path,title,limit=-1):
	spikes = utils.read_spike_events(path)
	isis = utils.get_ISI(spikes)
	if limit >0:
		isis_y = isis[1:limit+1]
	else:
		isis_y = isis[1:]
		# limit = max(isis)
	print(limit)
	# plt.plot(isis[:limit],isis_y,'.')
	plt.plot(isis[:-1],isis[1:],'.')

	if limit>0:
		plt.xlim(-5,limit)
		plt.ylim(-5,limit)
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


# if(i==1):
# 	ax1 = plt.subplot(rows,1,i)
# else:	
# 	plt.subplot(rows,1,i,sharex=ax1)
plt.figure(figsize=(20,20))
ax2 = plt.subplot(3,2,1)
read_n_plot(path_control_pre,"control_pre",)
ax1 = plt.subplot(3,2,2)
read_n_plot(path_control_pre,"control_pre",limit=70)
plt.subplot(3,2,3,sharex=ax2,sharey=ax2)
read_n_plot(path_laser,"laser")
plt.subplot(3,2,4,sharex=ax1,sharey=ax1)
read_n_plot(path_laser,"laser",limit=70)
plt.subplot(3,2,5,sharex=ax2,sharey=ax2)
read_n_plot(path_control_pos,"control_pos")
plt.subplot(3,2,6,sharex=ax1,sharey=ax1)
read_n_plot(path_control_pos,"control_pos",limit=70)

plt.tight_layout()

if save:
	save_path = path+"_return_map_"
	print(save_path)
	plt.savefig(save_path+".eps",format='eps')
	plt.savefig(save_path+".png",format='png')

if show:
	plt.show()

# spikes = utils.read_spike_events(path)

# plt.plot(spikes)
# plt.show()

# isis = utils.get_ISI(spikes)

# plt.plot(isis[:-1],isis[1:],'.')
# plt.show()