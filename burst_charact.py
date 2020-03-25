
from charact_utils import *


N1m = 'N1M'
N2v = 'N2v'
N3t = 'N3t'


def trunc(values, decs=0):
    return np.trunc(values*10**decs)/(10**decs)



#####################
# Loads file from record
date = '08-Jul-2019'
file = 'burst_ph1.txt'
file_name = '08-Jul-2019'

path = '../data/'+date+'/'

N1m_data = read_bursts_events(path+file)

# date = '08-Jul-2019'
file = 'burst_ph2.txt'

N2v_data = read_bursts_events(path+file)

# date = '08-Jul-2019'
file = 'burst_ph3.txt'

N3t_data = read_bursts_events(path+file)


print(len(N1m_data),len(N2v_data),len(N3t_data))
#Adjust burst to periods
#Input: Ref, Snd, Thrd.

N1m_data,N2v_data,N3t_data = fix_length(N1m_data,N2v_data,N3t_data)

print(len(N1m_data),len(N2v_data),len(N3t_data))


ran_x = (0,90)
ran_y = (0,90)
# #####################

# ####################
# ## Loads file from model neuron

# ### test2: n3t [1:] on load
# ##			n1m, n2v [:-1] on plot

# ##  test3: n3t [1:] on load
# ##			n1m, n2v, n3t [:-1] on plot

# # path = ""


# # # path = "../model/slow-fast/" + file_name + "/"
# # # path = "../lymnaea-model/test_intervals_long/"
# # # path = file_name +"/"
# path = "../model/no_variability/"

# # path = "../model/SO/"
# path = "../model/N1m/"
# # path = "../model/N3t/test17/"


# if len(sys.argv) >2:
# 	path = sys.argv[1]
# 	file_name = sys.argv[2]
# 	time_scale= int(sys.argv[3])
# else:
# 	file_name = "so_test19"
# 	# file_name = "test1"
# 	# file_name = "n3t_test17"
# 	# time_scale = 10000
# 	time_scale = 1000
# 	# time_scale = 1000


# # python3 burst_charact.py ../model/N3t/test17/ n3t_test17 1000; python3 burst_charact.py ../model/SO/ so_test19 10000;python3 burst_charact.py ../model/N1m/ test1 1000



# path+= file_name + "/"


# N1m_data = read_model_burst_path(path+"N1M",scale=time_scale)
# N2v_data = read_model_burst_path(path+"N2v",scale=time_scale)
# N3t_data = read_model_burst_path(path+"N3t",scale=time_scale)


# print("-------",len(N1m_data),len(N2v_data),len(N3t_data))
# # N1m_data = N1m_data[:len(N1m_data)//2]
# # N2v_data = N2v_data[:len(N2v_data)//2]
# # N3t_data = N3t_data[:len(N3t_data)//2]


# print(len(N1m_data),len(N2v_data),len(N3t_data))
# # #Adjust burst to periods
# # #Input: Ref, Snd, Thrd.

# N1m_data,N2v_data,N3t_data = fix_length(N1m_data,N2v_data,N3t_data)


# ran_x = (1.5,5.0)
# ran_y = (0,5.0)
###########################################################################



# print(N1m_data.shape)
# N1m_data=np.around(N1m_data,decimals=2)
# N2v_data=np.around(N2v_data,decimals=2)
# N3t_data=np.around(N3t_data,decimals=2)
# print(N1m_data[:2])


N1m_data=trunc(N1m_data,decs=2)
N2v_data=trunc(N2v_data,decs=2)
N3t_data=trunc(N3t_data,decs=2)
print(N1m_data[:2])

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


save=True
show=False




plot_intervals_stats(stats)
# 
if save:
	plt.savefig("./results_invariant_tests/images/"+file_name+"_boxplot.png")
# 
if show:
	plt.show()

# plot_intervals_stats(stats,norm=True)
# plt.savefig("./results_invariant_tests/images/"+file_name+"_boxplot_norm.png")
# plt.show()

# plot_bar(stats)
# plt.savefig("./results_invariant_tests/images/"+file_name+"_std_bar.png")
# plt.show()

# plot_intervals_stats(stats,norm=True,pos=True)
# plt.savefig("./results_invariant_tests/images/"+file_name+"_boxplot_norm_pos.png")
# plt.show()
# colors = ['maroon', 'teal', 'brown', 'blue', 'green']




# print("\n","\n",stats,"\n","\n")

# # print(N1m_data.shape,N2v_data.shape,N3t_data.shape)
# # # print(N1m_interv.shape,N2v_interv.shape,N3t_interv.shape)

# # #TODO: if shapes != then reshape(min(shape))


#####################



# ####################################
# ########  CORRELATIONS  ############
# ####################################

period = N1m_interv[PER]
print(period.shape,N3t_interv[DUR].shape)

plt.figure(figsize=(20,5))

plt.subplot(1,3,1)
plot_corr(period,N1m_interv[DUR][:-1],"Period (s)",N1m +" Burst Duration"+" (s)",ran_x,ran_y,False,color='blue')

plt.subplot(1,3,2)
plot_corr(period,N2v_interv[DUR][:-1],"Period (s)",N2v+" Burst Duration" +" (s)",ran_x,ran_y,False,color='blue')

plt.subplot(1,3,3)
plot_corr(period,N3t_interv[DUR][:-1],"Period (s)",N3t+" Burst Duration" +" (s)",ran_x,ran_y,False,color='blue')


plt.tight_layout()
if save:
	plt.savefig("./results_invariant_tests/images/cns_"+file_name+".png")
# 
if show:
	plt.show()


# # period = N1m_interv[PER]
# # print(period.shape,N3t_interv[DUR].shape)

# # plt.figure(figsize=(20,5))
# # ran_x = (0,90.0)
# # plt.subplot(1,3,1)
# # plot_corr(period,N1m_interv[IBI],"Period (s)","IBI "+N1m +" (s)",ran_x,ran_y,False,color='blue')

# # plt.subplot(1,3,2)
# # plot_corr(period,N2v_interv[IBI],"Period (s)","IBI "+N2v +" (s)",ran_x,ran_y,False,color='blue')

# # plt.subplot(1,3,3)
# # plot_corr(period,N3t_interv[IBI],"Period (s)","IBI "+N3t +" (s)",ran_x,ran_y,False,color='blue')


# # plt.tight_layout()
# # # plt.savefig("./results_invariant_tests/images/"+file_name+".png")
# # plt.show()




period = N1m_interv[PER]

plt.figure(figsize=(20,10))
#ran_x = (-1,90.0)
plt.subplot(2,3,1)
plot_corr(period,N1N2[INTERVAL][:-1],"Period (s)","N1-N2 interval (s)",ran_x,ran_y,False,color='green')

plt.subplot(2,3,4)
plot_corr(period,N2N1[INTERVAL],"Period (s)","N2-N1 interval (s)",ran_x,ran_y,False,color='green')

plt.subplot(2,3,2)
plot_corr(period,N1N3[INTERVAL][:-1],"Period (s)","N1-N3 interval (s)",ran_x,ran_y,False,color='green')

plt.subplot(2,3,5)
plot_corr(period,N3N1[INTERVAL],"Period (s)","N3-N1 interval (s)",ran_x,ran_y,False,color='green')

plt.subplot(2,3,3)
plot_corr(period,N2N3[INTERVAL][:-1],"Period (s)","N2-N3 interval (s)",ran_x,ran_y,False,color='green')

plt.subplot(2,3,6)
plot_corr(period,N3N2[INTERVAL],"Period (s)","N3-N2 interval (s)",ran_x,ran_y,False,color='green')


plt.tight_layout()
# 
if save:
	plt.savefig("./results_invariant_tests/images/"+file_name+"_intervals.png")
# 
if show:
	plt.show()


period = N1m_interv[PER]

plt.figure(figsize=(20,10))
ran_x = (1.5,4.0)
ran_y = (-0.95,5.0)
plt.subplot(2,3,1)
plot_corr(period,N1N2[DELAY][:-1],"Period (s)","N1-N2 delay (s)",ran_x,ran_y,False,color='brown')

plt.subplot(2,3,4)
plot_corr(period,N2N1[DELAY],"Period (s)","N2-N1 delay (s)",ran_x,ran_y,False,color='brown')

plt.subplot(2,3,2)
plot_corr(period,N1N3[DELAY][:-1],"Period (s)","N1-N3 delay (s)",ran_x,ran_y,False,color='brown')

plt.subplot(2,3,5)
plot_corr(period,N3N1[DELAY],"Period (s)","N3-N1 delay (s)",ran_x,ran_y,False,color='brown')

plt.subplot(2,3,3)
plot_corr(period,N2N3[DELAY][:-1],"Period (s)","N2-N3 delay (s)",ran_x,ran_y,False,color='brown')

plt.subplot(2,3,6)
plot_corr(period,N3N2[DELAY],"Period (s)","N3-N2 delay (s)",ran_x,ran_y,False,color='brown')



plt.tight_layout()
# 
if save:
	plt.savefig("./results_invariant_tests/images/"+file_name+"_delays.png")

if show:
	plt.show()
# 


