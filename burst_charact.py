
from charact_utils import *


N1m = 'N1M'
N2v = 'N2v'
N3t = 'N3t'





#####################
# ## Loads file from record
# date = '08-Jul-2019'
# file = 'burst_ph1.txt'
# file_name = '08-Jul-2019'

# N1m_data = read_bursts_events('../'+date+'/'+file)

# date = '08-Jul-2019'
# file = 'burst_ph2.txt'

# N2v_data = read_bursts_events('../'+date+'/'+file)

# date = '08-Jul-2019'
# file = 'burst_ph3.txt'

# N3t_data = read_bursts_events('../'+date+'/'+file)


# print(len(N1m_data),len(N2v_data),len(N3t_data))
# #Adjust burst to periods
# #Input: Ref, Snd, Thrd.

# N1m_data,N2v_data,N3t_data = fix_length(N1m_data,N2v_data,N3t_data)

# print(len(N1m_data),len(N2v_data),len(N3t_data))

# N1m_interv = analyse(N1m_data,N1m)

# # N2v_data = N2v_data
# N2v_interv = analyse(N2v_data,N2v)


# N3t_interv = analyse(N3t_data,N3t)

# #####################

####################
### Loads file from model neuron

#### test2: n3t [1:] on load
###			n1m, n2v [:-1] on plot

###  test3: n3t [1:] on load
###			n1m, n2v, n3t [:-1] on plot

path = ""


if len(sys.argv) >1:
	file_name = sys.argv[1]
else:
	file_name = "test1"

# path = "../model/slow-fast/" + file_name + "/"
path = "../lymnaea-model/test_intervals_long/"
# path = file_name +"/"
# path = "../model/N1m/test1/"

N1m_data = read_model_burst_path(path+N1m,scale=1)
N2v_data = read_model_burst_path(path+N2v,scale=1)
N3t_data = read_model_burst_path(path+N3t,scale=1)

# print(len(N1m_data),len(N2v_data),len(N3t_data))
# #Adjust burst to periods
# #Input: Ref, Snd, Thrd.

N1m_data,N2v_data,N3t_data = fix_length(N1m_data,N2v_data,N3t_data)


N1m_interv = analyse(N1m_data,N1m,plot=False)
N2v_interv = analyse(N2v_data,N2v,plot=False)
N3t_interv = analyse(N3t_data,N3t,plot=False)


# # print(N1m_data.shape,N2v_data.shape,N3t_data.shape)
# # # print(N1m_interv.shape,N2v_interv.shape,N3t_interv.shape)

# # #TODO: if shapes != then reshape(min(shape))


#####################


N1N2,N2N1 = get_intervals(N1m_data,N2v_data)
N1N3,N3N1 = get_intervals(N1m_data,N3t_data)
N2N3,N3N2 = get_intervals(N2v_data,N3t_data)



####################################
########  CORRELATIONS  ############
####################################

period = N1m_interv[PER]
print(period.shape,N3t_interv[DUR].shape)

plt.figure(figsize=(20,5))
ran = (0,100.0)
plt.subplot(1,3,1)
plot_corr(period,N1m_interv[DUR][:-1],"Period (s)","Duration of "+N1m +" (s)",ran,False)

plt.subplot(1,3,2)
plot_corr(period,N2v_interv[DUR][:-1],"Period (s)","Duration of "+N2v +" (s)",ran,False)

plt.subplot(1,3,3)
plot_corr(period,N3t_interv[DUR][:-1],"Period (s)","Duration of "+N3t +" (s)",ran,False)


plt.savefig("./results_invariant_tests/images/"+file_name+".png")
plt.show()



period = N1m_interv[PER]

plt.figure(figsize=(20,10))
ran = (-1,100.0)
plt.subplot(2,3,1)
plot_corr(period,N1N2[INTERVAL][:-1],"Period (s)","N1-N2 interval (s)",ran,False)

plt.subplot(2,3,4)
plot_corr(period,N2N1[INTERVAL],"Period (s)","N2-N1 interval (s)",ran,False)

plt.subplot(2,3,2)
plot_corr(period,N1N3[INTERVAL][:-1],"Period (s)","N1-N3 interval (s)",ran,False)

plt.subplot(2,3,5)
plot_corr(period,N3N1[INTERVAL],"Period (s)","N3-N1 interval (s)",ran,False)

plt.subplot(2,3,3)
plot_corr(period,N2N3[INTERVAL][:-1],"Period (s)","N2-N3 interval (s)",ran,False)

plt.subplot(2,3,6)
plot_corr(period,N3N1[INTERVAL],"Period (s)","N3-N2 interval (s)",ran,False)



plt.savefig("./results_invariant_tests/images/"+file_name+"_intervals.png")
plt.show()


period = N1m_interv[PER]

plt.figure(figsize=(20,10))
ran = (-1,100.0)
plt.subplot(2,3,1)
plot_corr(period,N1N2[DELAY][:-1],"Period (s)","N1-N2 delay (s)",ran,False)

plt.subplot(2,3,4)
plot_corr(period,N2N1[DELAY],"Period (s)","N2-N1 delay (s)",ran,False)

plt.subplot(2,3,2)
plot_corr(period,N1N3[DELAY][:-1],"Period (s)","N1-N3 delay (s)",ran,False)

plt.subplot(2,3,5)
plot_corr(period,N3N1[DELAY],"Period (s)","N3-N1 delay (s)",ran,False)

plt.subplot(2,3,3)
plot_corr(period,N2N3[DELAY][:-1],"Period (s)","N2-N3 delay (s)",ran,False)

plt.subplot(2,3,6)
plot_corr(period,N3N1[DELAY],"Period (s)","N3-N1 delay (s)",ran,False)



plt.savefig("./results_invariant_tests/images/"+file_name+"_delays.png")
plt.show()

