
from charact_utils import *


#####################
### Loads file from file name

file_name = '../10-Jul-2019/b_burst.txt'
neuron1 = 'b'
# file_name = '../model/model_'+neuron1+'_burst.txt'

data1 = np.loadtxt(file_name)
data1 /= 1000


char1 = analyse_hists(data1,neuron1)



file_name2 = '../10-Jul-2019/b1_burst.txt'
neuron2 = 'b1'
# file_name = '../model/model_'+neuron+'_burst.txt'

data2 = np.loadtxt(file_name2)
data1 /= 1000


char2 = analyse_hists(data2,neuron2)


###################################################
######## 2 burst plot 
################################################
# plt.subplot(2,3,1)
# plt.title("Burst duration" + neuron1)
# plt.hist(dur1,rwidth=0.4)
# plt.xlabel("Time (s)")


# plt.subplot(2,3,2)
# plt.title("Burst Interval" + neuron1)
# plt.hist(ibi1,rwidth=0.4)
# plt.xlabel("Time (s)")

# plt.subplot(2,3,3)
# plt.title("Burst Period" + neuron1)
# plt.hist(per1,rwidth=0.4)
# plt.xlabel("Time (s)")


# plt.subplot(2,3,4)
# plt.title("Burst duration" + neuron1)
# plt.hist(dur2,rwidth=0.4)
# plt.xlabel("Time (s)")


# plt.subplot(2,3,5)
# plt.title("Burst Interval" + neuron1)
# plt.hist(ibi2,rwidth=0.4)
# plt.xlabel("Time (s)")

# plt.subplot(2,3,6)
# plt.title("Burst Period" + neuron1)
# plt.hist(per2,rwidth=0.4)
# plt.xlabel("Time (s)")

# plt.show()



####################################
########  CORRELATIONS  ############
####################################

plot_corr(char1[IBI],char1[PER],"IBI","Period")
plot_corr(char1[PER],char1[PER],"Period","Duration")

# plt.plot(ibi1,per1,'.')
# plt.xlabel("Period")
# plt.ylabel("IBI")
# plt.show()

# plt.plot(per1,dur1[:-1],'.')
# plt.xlabel("Period")
# plt.ylabel("Duration")
# plt.show()