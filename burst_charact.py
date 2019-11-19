
from charact_utils import *


N1m = 'N1M'
N2v = 'N2v'
N3t = 'N3t'





#####################
## Loads file from record
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


# stats={}
# N1m_interv = analyse(N1m_data,N1m,stats)

# # N2v_data = N2v_data
# N2v_interv = analyse(N2v_data,N2v,stats)


# N3t_interv = analyse(N3t_data,N3t,stats)

#####################

####################
### Loads file from model neuron

#### test2: n3t [1:] on load
###			n1m, n2v [:-1] on plot

###  test3: n3t [1:] on load
###			n1m, n2v, n3t [:-1] on plot

path = ""


# path = "../model/slow-fast/" + file_name + "/"
# path = "../lymnaea-model/test_intervals_long/"
# path = file_name +"/"
path = "../model/N1m/test1/"


if len(sys.argv) >2:
	path = sys.argv[1]
	file_name = sys.argv[2]
else:
	file_name = "test1"


N1m_data = read_model_burst_path(path+N1m,scale=100)
N2v_data = read_model_burst_path(path+N2v,scale=100)
N3t_data = read_model_burst_path(path+N3t,scale=100)

print(len(N1m_data),len(N2v_data),len(N3t_data))
# #Adjust burst to periods
# #Input: Ref, Snd, Thrd.

N1m_data,N2v_data,N3t_data = fix_length(N1m_data,N2v_data,N3t_data)






stats={}

index = [0]

N1m_interv = analyse(N1m_data,N1m,stats,index)
N2v_interv = analyse(N2v_data,N2v,stats,index)
N3t_interv = analyse(N3t_data,N3t,stats,index)



N1N2,N2N1 = analyse_pair(N1m_data,N2v_data,"N1","N2",stats,index)
N1N3,N3N1 = analyse_pair(N1m_data,N3t_data,"N1","N3",stats,index)
N2N3,N3N2 = analyse_pair(N2v_data,N3t_data,"N2","N3",stats,index)
print(index)

# print("\n","\n",stats,"\n","\n")



plot_intervals_stats(stats)
plt.savefig("./results_invariant_tests/images/"+file_name+"_boxplot.png")
plt.show()

# colors = ['maroon', 'teal', 'brown', 'blue', 'green']




# print("\n","\n",stats,"\n","\n")

# # print(N1m_data.shape,N2v_data.shape,N3t_data.shape)
# # # print(N1m_interv.shape,N2v_interv.shape,N3t_interv.shape)

# # #TODO: if shapes != then reshape(min(shape))


#####################



####################################
########  CORRELATIONS  ############
####################################

period = N1m_interv[PER]
print(period.shape,N3t_interv[DUR].shape)

plt.figure(figsize=(20,5))
ran = (0,90.0)
plt.subplot(1,3,1)
plot_corr(period,N1m_interv[DUR][:-1],"Period (s)","Duration of "+N1m +" (s)",ran,False,color='blue')

plt.subplot(1,3,2)
plot_corr(period,N2v_interv[DUR][:-1],"Period (s)","Duration of "+N2v +" (s)",ran,False,color='blue')

plt.subplot(1,3,3)
plot_corr(period,N3t_interv[DUR][:-1],"Period (s)","Duration of "+N3t +" (s)",ran,False,color='blue')


plt.tight_layout()
# plt.savefig("./results_invariant_tests/images/"+file_name+".png")
# plt.show()


# period = N1m_interv[PER]
# print(period.shape,N3t_interv[DUR].shape)

# plt.figure(figsize=(20,5))
# ran = (0,90.0)
# plt.subplot(1,3,1)
# plot_corr(period,N1m_interv[IBI],"Period (s)","IBI "+N1m +" (s)",ran,False,color='blue')

# plt.subplot(1,3,2)
# plot_corr(period,N2v_interv[IBI],"Period (s)","IBI "+N2v +" (s)",ran,False,color='blue')

# plt.subplot(1,3,3)
# plot_corr(period,N3t_interv[IBI],"Period (s)","IBI "+N3t +" (s)",ran,False,color='blue')


# plt.tight_layout()
# # plt.savefig("./results_invariant_tests/images/"+file_name+".png")
# plt.show()




period = N1m_interv[PER]

plt.figure(figsize=(20,10))
ran = (-1,90.0)
plt.subplot(2,3,1)
plot_corr(period,N1N2[INTERVAL][:-1],"Period (s)","N1-N2 interval (s)",ran,False,color='green')

plt.subplot(2,3,4)
plot_corr(period,N2N1[INTERVAL],"Period (s)","N2-N1 interval (s)",ran,False,color='green')

plt.subplot(2,3,2)
plot_corr(period,N1N3[INTERVAL][:-1],"Period (s)","N1-N3 interval (s)",ran,False,color='green')

plt.subplot(2,3,5)
plot_corr(period,N3N1[INTERVAL],"Period (s)","N3-N1 interval (s)",ran,False,color='green')

plt.subplot(2,3,3)
plot_corr(period,N2N3[INTERVAL][:-1],"Period (s)","N2-N3 interval (s)",ran,False,color='green')

plt.subplot(2,3,6)
plot_corr(period,N3N1[INTERVAL],"Period (s)","N3-N2 interval (s)",ran,False,color='green')


plt.tight_layout()
# plt.savefig("./results_invariant_tests/images/"+file_name+"_intervals.png")
# plt.show()


period = N1m_interv[PER]

plt.figure(figsize=(20,10))
ran = (-1,90.0)
plt.subplot(2,3,1)
plot_corr(period,N1N2[DELAY][:-1],"Period (s)","N1-N2 delay (s)",ran,False,color='brown')

plt.subplot(2,3,4)
plot_corr(period,N2N1[DELAY],"Period (s)","N2-N1 delay (s)",ran,False,color='brown')

plt.subplot(2,3,2)
plot_corr(period,N1N3[DELAY][:-1],"Period (s)","N1-N3 delay (s)",ran,False,color='brown')

plt.subplot(2,3,5)
plot_corr(period,N3N1[DELAY],"Period (s)","N3-N1 delay (s)",ran,False,color='brown')

plt.subplot(2,3,3)
plot_corr(period,N2N3[DELAY][:-1],"Period (s)","N2-N3 delay (s)",ran,False,color='brown')

plt.subplot(2,3,6)
plot_corr(period,N3N2[DELAY],"Period (s)","N3-N2 delay (s)",ran,False,color='brown')



plt.tight_layout()
# plt.savefig("./results_invariant_tests/images/"+file_name+"_delays.png")
# plt.show()



