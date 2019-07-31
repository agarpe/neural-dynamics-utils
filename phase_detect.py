#phase_detect.py
#Phase detection from spike events detected. Detecting change in spikes frequency by SDF.

from charact_utils import *



#####################
### Loads file from file name

# file_name = '../24-Jun-2019/24-Jun_N1m-spikes.txt'
neuron = 'B4'
# file_name = '../model/model_'+neuron+'_spikes.txt'
file_name = '../08-Jul-2019/spikes_b4.txt'

data_n = np.loadtxt(file_name)

print(data_n.shape)

#Change to secs

data_n /= 1000

#get half

data_n = data_n[:(data_n.shape[0]//1)]


mean_evt_n = to_mean(data_n)
print(mean_evt_n.shape)
print(mean_evt_n[:4],mean_evt_n[-1])
#######################################################
spikes = get_spikes(mean_evt_n)


sdf_2 = sdf(spikes,7000,500)

# plt.plot(sdf_2)
# plt.ylabel("SDF")
# plt.xlabel("Time")
# plt.show()


# plt.plot(spikes,'.')
# plt.plot(sdf_2)
# plt.show()

### saving sdf signal 

# f1 = open("sdf.txt","w")
# np.savetxt(f1,sdf_2,delimiter='\t')
# print(sdf_2.shape)
# f1.close()


# sdf_2 *= 0.001


####
ev = events_from_thres(sdf_2)
print("EV",ev.shape)
plt.plot(sdf_2)
plt.plot(ev[:,0],ev[:,1],'.')

plt.show()


# ev *= 10


#####
phs = get_phases_from_events(ev[:,0])
ph1 = np.array(phs[0])
ph2 = np.array(phs[1])
ph3 = np.array(phs[2])
print(ph1.shape,ph1[0])
print(ph2.shape,ph2[0])
print(ph3.shape,ph3[0])

plt.plot(sdf_2)
plt.plot(ev[:,0],ev[:,1],'.')
plt.plot(ph1,np.zeros(ph1.shape),'.')
plt.plot(ph2,np.ones(ph2.shape),'.')
plt.plot(ph3,np.ones(ph3.shape)+1,'.')
plt.show()


save_events(ph1,file_name[:-4]+"-ph1.txt")
save_events(ph2,file_name[:-4]+"-ph2.txt")
save_events(ph3,file_name[:-4]+"-ph3.txt")
save_events(sdf_2,file_name[:-4]+"-events.txt")





# #######################################################
# ## DIFF 

# diff = np.diff(sdf_2)
# plt.plot(diff)
# plt.show()

# plt.ylim(0,5)
# plt.plot(spikes,'.')
# plt.plot(diff)
# plt.show()



# plt.plot(sdf_2)
# plt.plot(np.diff(sdf_2))
# plt.show()

# ax1=plt.subplot(2,1,1)
# plt.plot(sdf_2)
# plt.subplot(2,1,2,sharex = ax1)
# plt.plot(diff)
# plt.show()

# plt.plot(np.diff(sdf_2,2))
# plt.show()

# #########################################################

# ##### PHASES PARTITION


# # init = 479200
# # end = 745200
# # ph1,ph2 = get_phases(sdf_2[init:end],init,end)
# # print(ph1)
# # plt.plot(sdf_2)
# # plt.plot(ph1[:,0],ph1[:,1],'.')
# # plt.plot(ph2[:,0],ph2[:,1],'.')
# # plt.ylabel("SDF")
# # plt.xlabel("Time")
# # plt.show()


# # f1 = open("events_2_p1.txt",'w')
# # f2 = open("events_2_p2.txt",'w')
# # print(ph1[:10,0])
# # time = ph1[:,0]
# # time = time.reshape((time.shape[0]//2,2))
# # time2 = ph2[:,0]
# # time2 = time2.reshape((time2.shape[0]//2,2))


# # time3 = [time2[:,1]time1[1:,0]]
# # time2 = time2.reshape((time2.shape[0]//2,2))



# # np.savetxt(f1,time,delimiter='\t')
# # print(time.shape)
# # f1.close()
# # np.savetxt(f2,time2,delimiter='\t')
# # f2.close()



# #########################################3
# ######### v2
# init = 0
# end = diff.shape[0]
# ph1,ph2 = get_phases(diff[init:end],init,end,0.0025,-0.0044)
# print(ph1)
# print(ph1.shape,ph2.shape)

# plt.plot(diff)
# plt.plot(ph1[:,0],ph1[:,1],'.')
# plt.plot(ph2[:,0],ph2[:,1],'.')
# plt.ylabel("DIFF")
# plt.xlabel("Time")
# plt.show()

# plt.plot(sdf_2)
# plt.plot(ph1[:,0],ph1[:,1],'.')
# plt.plot(ph2[:,0],ph2[:,1],'.')
# plt.ylabel("SDF")
# plt.xlabel("Time")
# plt.show()


# f1 = open("events_2_p1.txt",'w')
# f2 = open("events_2_p2.txt",'w')
# print(ph1[:10,0])
# time = ph1[:,0]
# time = time.reshape((time.shape[0]//2,2))
# time2 = ph2[:,0]
# time2 = time2.reshape((time2.shape[0]//2,2))


# # time3 = [time2[:,1]time1[1:,0]]
# # time2 = time2.reshape((time2.shape[0]//2,2))



# np.savetxt(f1,time,delimiter='\t')
# print(time.shape)
# f1.close()
# np.savetxt(f2,time2,delimiter='\t')
# f2.close()

#########################################################
# plt.subplot(2,1,1)
# plt.plot(sdf_2)
# plt.ylabel("SDF")
# plt.xlabel("Time")

# sdf_2 = sdf(spikes,3,5)
# plt.subplot(2,1,2)
# plt.plot(sdf_2)
# plt.ylabel("SDF")
# plt.xlabel("Time")

# plt.show()


###################################################
##### PARTITION BY THRESHOLD
##################################################

# phase1 = diff[np.where(diff >= 0.0025 )]

# phase2 = diff[np.where(diff < -0.0044)]


# # phase1 = events_from_thres(diff, 0.0025)
# # phase2 = events_from_thres(diff, -0.0044)


# plt.subplot(3,1,1)
# plt.plot(sdf_2)
# plt.subplot(3,1,2)
# plt.plot(phase1,'.')
# plt.subplot(3,1,3)
# plt.plot(phase2,'.')
# plt.show()

# plt.plot(mean_evt_n,np.full(mean_evt_n.shape,1),'.')
# plt.show()



# phase11 = ISI_n[np.where(sdf_[:2000] < 0.85)]
# phase12 = ISI_n[np.where(sdf_[2000:] < 0.7)]
# # phase11 = ISI_n[np.where(sdf_[:2000] < 0.7)]
# phase1 = np.concatenate((phase11,phase12))

# phase21 = ISI_n[np.where(sdf_[:2000] > 0.85)]
# phase22 = ISI_n[np.where(sdf_[2000:] > 0.7)]
# phase2 = np.concatenate((phase21,phase22))

# plt.subplot(3,1,1)
# plt.plot(ISI_n,'.')
# plt.subplot(3,1,2)
# plt.plot(phase1,'.')
# plt.subplot(3,1,3)
# plt.plot(phase2,'.')
# plt.show()

# plt.plot(mean_evt_n,np.full(mean_evt_n.shape,1),'.')
# plt.show()
