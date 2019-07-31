
from charact_utils import *
#####################
### Loads file from file name

# file_name = '../10-Jul-2019/b_burst.txt'
neuron1 = 'N1M'
file_name = '../model/model_'+neuron1+'_burst.txt'

data1 = np.loadtxt(file_name)


# file_name = '../10-Jul-2019/b1_burst.txt'
# neuron2 = 'b1'
# file_name = '../model/model_'+neuron+'_burst.txt'

data2 = np.loadtxt(file_name)


top = len(data1)
# top = -5
# top = 90
data1 = data1[:top]

print(data1.shape)
#Change to secs

data1 /= 1000


dur1,ibi1,per1 = analyse(data1)

# dur2,ibi2,per2 = analyse(data2)



# dur = get_burst_duration(data_n1m)

# ibi = get_burst_interval(data_n1m)

# period = get_burst_period(data_n1m)

plt.subplot(1,3,1)
plt.title("Burst duration" + neuron1)
plt.hist(dur1,rwidth=0.4)
plt.xlabel("Time (s)")


plt.subplot(1,3,2)
plt.title("Burst Interval" + neuron1)
plt.hist(ibi1,rwidth=0.4)
plt.xlabel("Time (s)")

plt.subplot(1,3,3)
plt.title("Burst Period" + neuron1)
plt.hist(per1,rwidth=0.4)
plt.xlabel("Time (s)")
plt.show()


# plt.bar(range(len(dur)),dur)
# plt.show()
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

plt.plot(ibi1,per1,'.')
plt.xlabel("Period")
plt.ylabel("IBI")
plt.show()


plt.plot(per1,dur1[:-1],'.')
plt.xlabel("Period")
plt.ylabel("Duration")
plt.show()