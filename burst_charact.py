import charact_utils as utils 
import matplotlib.pyplot as plt
import argparse

DUR =0
IBI =1


def plot_hist(data,title,xlabel="Time (ms)",rang=None,width=10):
	plt.hist(data,rang,width=width)
	plt.title(title)
	plt.xlabel(xlabel)

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=True, help="Path to the file with the spike events")
ap.add_argument("-r","--range",required=False,default=None, help="Range of data")
ap.add_argument("-sh", "--show", required=False,default='y', help="Show plot")
ap.add_argument("-sv", "--save", required=False,default='y', help="Save events")
ap.add_argument("-ex", "--ext", required=False, default='', help="Extension after laser or control.")

args = vars(ap.parse_args())
ext = args['ext']

if ext != '':
	ext += '_'

path = args['path']
path1 = path+"_control_pre_"+ext+"_burst_events.txt"
path2 = path+"_laser_"+ext+"_burst_events.txt"
path3 = path+"_control_pos_"+ext+"_burst_events.txt"

rang = args['range']
rang_str = ''
if str(rang) != 'None':
	rang = [float(r) for r in rang.split()]
	rang_str = "range"

show= True if args['show']=='y' else False 
save= True if args['save']=='y' else False 


burst1 = utils.read_bursts_events(path1)
burst2 = utils.read_bursts_events(path2)
burst3 = utils.read_bursts_events(path3)


stats1 = utils.get_single_intervals(burst1)
stats2 = utils.get_single_intervals(burst2)
stats3 = utils.get_single_intervals(burst3)


plt.figure(figsize=(20,20))
ax2 = plt.subplot(3,2,1)
plot_hist(stats1[DUR],"Control pre",xlabel="Burst duration (ms)",width=10,rang=rang)
ax1 = plt.subplot(3,2,2)
plot_hist(stats1[IBI],"Control pre",xlabel="IBI (ms)",width=100,rang=rang)
plt.subplot(3,2,3,sharex=ax2,sharey=ax2)
plot_hist(stats2[DUR],"Laser",xlabel="Burst duration (ms)",width=10,rang=rang)
plt.subplot(3,2,4,sharex=ax1,sharey=ax1)
plot_hist(stats2[IBI],"Laser",xlabel="IBI (ms)",width=100,rang=rang)
plt.subplot(3,2,5,sharex=ax2,sharey=ax2)
plot_hist(stats3[DUR],"Control pos",xlabel="Burst duration (ms)",width=10,rang=rang)
plt.subplot(3,2,6,sharex=ax1,sharey=ax1)
plot_hist(stats3[IBI],"Control pos",xlabel="IBI (ms)",width=100,rang=rang)

plt.tight_layout()


if save:
	save_path = path+"_burst_stats_"+rang_str
	print(save_path)
	plt.savefig(save_path+".eps",format='eps')
	plt.savefig(save_path+".png",format='png')
if show:
	plt.show()

