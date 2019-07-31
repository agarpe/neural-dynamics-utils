#isi_charact.py
#Caracterization of neural activity from spikes events, based on spikes intervals and frequencies. 

from charact_utils import *



#####################
### Loads file from file name

file_name = '../24-Jun-2019/24-Jun_N1m-spikes.txt'

# file_name = '../model/model_'+neuron+'_spikes.txt'
# file_name = '../08-Jul-2019/spikes_b4.txt'

#Posible neuron name
neuron = 'N1m'


mean_evt_n = read_spike_events(file_name)

##########################################
####	ISI
#######################


ISI_n = get_ISI(mean_evt_n)

# plt.plot(ISI_n,'.')
# plt.show()

#Removing IBIs
# ISI_n = np.array(ISI_n)
# ISI_n = ISI_n[np.where(ISI_n < 2)]


sdf_ = sdf(ISI_n)


# plt.hist(ISI_n,rwidth=0.4,range=(0,1))
# plt.show()






########################
####	Plot
#######################
plt.subplot(2,1,1)
plt.plot(ISI_n,'.')
plt.ylabel("ISI")
plt.xlabel("Time")

plt.subplot(2,1,2)
plt.plot(sdf_)
plt.ylabel("SDF")
plt.xlabel("Time")
plt.show()
# plt.bar(range(len(ISI_n)),ISI_n)
# plt.show()

plt.hist(ISI_n,rwidth=0.4,range=(0,1))
plt.xlabel("ISI")
plt.ylabel("Freq")
plt.title("ISI histogram for "+neuron)
plt.show()

plot_return_map(ISI_n[:],neuron ,xlim=(-0.05,0.3),ylim=(-0.05,0.3))