#isi_charact.py
#Caracterization of neural activity from spikes events, based on spikes intervals and frequencies. 

from charact_utils import *
import sys


#####################

if len(sys.argv) > 1:
	print (sys.argv)
	if sys.argv[1] == "model":
		neuron = sys.argv[2]
		file_name = '../model/'+neuron+'_spikes.txt'
	else: #from file not model
		file_name = sys.argv[1]
		neuron = file_name

else:
	#Posible neuron name
	neuron = 'N1m'
	### Loads file from file name

	# file_name = '../24-Jun-2019/24-Jun_N1m-spikes.txt'

	file_name = '../model/'+neuron+'_spikes.txt'
	# file_name = '../08-Jul-2019/spikes_b4.txt'



mean_evt_n = read_spike_events(file_name)

##########################################
####	ISI
#######################


ISI_n = np.array(get_ISI(mean_evt_n))
spikes = get_spikes(mean_evt_n,0.01)
print(len(spikes))

# plt.plot(spikes,'.')
# plt.show()

print("Firing rate",len(mean_evt_n)/len(spikes))

ISI_n /= 100
# spikes /= 1000
# plt.plot(ISI_n,'.')
# plt.show()

#Removing IBIs
# ISI_n = np.array(ISI_n)
# ISI_n = ISI_n[np.where(ISI_n < 2)]


sdf_ISI = sdf(ISI_n)


sdf_spike = sdf(spikes)


print(np.std(sdf_ISI))
print(np.average(sdf_ISI))




########################
####	Plot
#######################


# plt.hist(spikes,rwidth=0.4,range=(0,0.1))
# plt.xlabel("ISI")
# plt.ylabel("Freq")
# plt.title("ISI histogram for "+neuron)
# plt.show()

# plt.subplot(2,1,1)
# plt.plot(ISI_n,'.')
# plt.ylabel("ISI")
# plt.xlabel("Time")

# plt.subplot(2,1,2)
# plt.plot(sdf_ISI)
# plt.ylabel("SDF from ISI")
# plt.xlabel("Time")
# plt.show()
# # plt.bar(range(len(ISI_n)),ISI_n)
# # plt.show()


# plt.plot(sdf_ISI)
# plt.ylabel("SDF from ISI")
# plt.xlabel("Time")
# plt.show()


# plt.plot(sdf_spike)
# plt.ylabel("SDF from spikes")
# plt.xlabel("Time")
# plt.show()


plt.hist(ISI_n,rwidth=1)
plt.xlabel("ISI")
plt.ylabel("Freq")
plt.title("ISI histogram for "+neuron)
plt.show()

plt.hist(ISI_n,rwidth=0.4,range=(0,0.3))
plt.xlabel("ISI")
plt.ylabel("Freq")
plt.title("ISI histogram for "+neuron)
plt.show()

plot_return_map(ISI_n[:],neuron ,xlim=(-0.05,0.3),ylim=(-0.05,0.3))


