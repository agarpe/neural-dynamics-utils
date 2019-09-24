
from charact_utils import *

N1m = 'N1m'
N2v = 'N2v'
N3t = 'N3t'


#####################
### Loads file from record
# date = '08-Jul-2019'
# file = 'burst_ph1.txt'

# N1m_data = read_bursts_events('../'+date+'/'+file)
# N1m_interv = analyse(N1m_data,N1m)


# date = '08-Jul-2019'
# file = 'burst_ph2.txt'

# N2v_data = read_bursts_events('../'+date+'/'+file)
# # N2v_data = N2v_data
# N2v_interv = analyse(N2v_data,N2v)


# date = '08-Jul-2019'
# file = 'burst_ph3.txt'

# N3t_data = read_bursts_events('../'+date+'/'+file)[1:]
# N3t_interv = analyse(N3t_data,N3t)

#####################

#####################
### Loads file from model neuron

N1m = 'N1m'
N1m_data = read_model_burst(N1m)
N1m_interv = analyse(N1m_data,N1m)


N2v = 'N2v'
N2v_data = read_model_burst(N2v)
N2v_interv = analyse(N2v_data,N2v)



N3t = 'N3t'
N3t_data = read_model_burst(N3t)
N3t_interv = analyse(N3t_data,N3t)



# print(N1m_data.shape,N2v_data.shape,N3t_data.shape)
# # print(N1m_interv.shape,N2v_interv.shape,N3t_interv.shape)

# #TODO: if shapes != then reshape(min(shape))


#####################

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
######## STATS #####################
####################################




####################################
########  CORRELATIONS  ############
####################################

# plot_corr(char1[PER],char2[IBI],"Period","Duration of "+neuron2)
# plot_corr(char2[PER],char2[PER],"Period","Duration")

period = N1m_interv[PER]


ran = (0.8,0.9)
plt.subplot(1,3,1)
# plt.ylim(0.1,1.2)
plt.ylim(ran)
plt.xlim(ran)
plot_corr(period,N1m_interv[DUR][:-1],"Period","Duration of "+N1m,False)
plt.subplot(1,3,2)
# plt.ylim(0.1,1.2)
plt.ylim(ran)
plt.xlim(ran)
plot_corr(period,N2v_interv[DUR],"Period","Duration of "+N2v,False)
plt.subplot(1,3,3)
# plt.ylim(0.1,1.2)
plt.ylim(ran)
plt.xlim(ran)
plot_corr(period,N3t_interv[DUR],"Period","Duration of "+N3t,False)
plt.show()


