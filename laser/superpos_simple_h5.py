
import h5py
import matplotlib.pyplot as plt
import numpy as np
import sys
import argparse 
from scipy.signal import find_peaks

colors=['b', 'r', 'g', 'k']

colors = plt.cm.tab20(np.linspace(0,1,10))


ap = argparse.ArgumentParser()

ap.add_argument("-p", "--path", required=True, help="Path to the file to analyze")
ap.add_argument("-c","--cols",required=False, default=None,help="Index to the elements in the trial separated by space")
ap.add_argument("-t","--trials",required=False, default=None,help="Selection of trials to plot")

args = vars(ap.parse_args())

path = args['path']
filename=path
file = path[path.rfind("/")+1:]
path = path[:path.rfind("/")+1]
print(path,file)

if args['cols'] is not None:
	columns = tuple([int(col) for col in args['cols'].split()])
	print("Selected columns", columns)
else:
	columns = args['cols']

if args['trials'] is not None:
	trials = tuple([int(trial) for trial in args['trials'].split()])
	print("Selected trials", columns)
else:
	trials = args['trials']


#Open file 
try:
	f = h5py.File(filename, 'r')		
except:
	print("File not valid")
	exit()


w_l= 300
w_r= 350


n_trial=1
while 1:
	
	trial = "Trial"+str(n_trial)
	struct = '/'+trial+'/Synchronous Data/Channel Data'

	try:
		dset = f[struct]
		data = dset[()]
	except KeyError as e:
		if 'Channel Data' in e.args[0]:
			print("Skiping Trial %d"%n_trial)
			n_trial+=1
			continue
		else:
			print("No trials left. %d files generated"%n_trial)
			break

	if trials is not None and n_trial not in trials:
		n_trial+=1
		continue

	print("Ploting trial", n_trial)

	try:
		signal = data[:,columns]
		signal = signal.reshape(signal.shape[0])
	except Exception as e:
		print("Warning: ", e.args)
		signal = np.array([])

	amp = np.max(signal) - np.min(signal)
	max_height = amp - amp*0.05
	max_height = np.max(signal) - amp*0.05
	# print(amp, max_height)

	spikes_t, spikes_v = find_peaks(signal, height=max_height, distance=100)

	# plt.figure()
	# plt.plot(signal)
	# plt.plot(spikes_t, signal[spikes_t], 'x')
	# plt.show()
	print(len(spikes_t))

	waveforms = np.array([signal[spike-w_l:spike+w_r] for spike in spikes_t[1:-1]])
	waveforms -= np.mean(waveforms)

	plt.plot(waveforms.T, color=colors[trials.index(n_trial)%len(trials)], alpha=0.2, linewidth=0.8)
	plt.plot(np.mean(waveforms, axis=0),linewidth=3, color=colors[trials.index(n_trial)], label=str(n_trial))
	plt.legend()

	n_trial+=1

	if n_trial > max(trials):
		break


plt.show()

