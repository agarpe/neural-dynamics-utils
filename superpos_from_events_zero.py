from charact_utils import *
import pandas as pd

if len(sys.argv) !=5:
	print("python3 superpos_from_events.py control_events_path laser_events_path")
	path = sys.argv[1]
	control_events_path = sys.argv[2]
	laser_events_path = sys.argv[3]
	time_scale=1

os.system("sed -i 's/\,/./g' "+path)
data =  pd.read_csv(path, delimiter = "\t",skiprows=4,header=None)

print(data.keys())
control_signal =data[0]
laser_signal =data[1]

# time = control_events[:]
# print(control_signal[0])
# print(data[0][0])

# control_events = read_bursts_events(control_events_path)
# laser_events = read_bursts_events(laser_events_path)

# print(control_events[0][0]/0.1)
counter =2
for i in range(control_signal.shape[0]):
	if(control_signal[i]==0 and counter!=0):
		counter = 0
	if(control_signal[i] != 0):
		plt.plot(counter,control_signal[i],color='b')
		counter+=1


plt.show()


for le in laser_events:
	plt.plot(laser_signal[le[0]:le[1]])

plt.show()