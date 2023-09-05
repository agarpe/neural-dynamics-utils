#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 16:57:38 2022

@author: rlevi
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path='/home/agarpe/Workspace/data/laser/pipette_temperature/14-09-2022/17h33m37s_Trial8_pipette_temperature_thermistor_14-09-22.asc'
#path='/home/agarpe/Workspace/data/laser/pipette_temperature/16h39m28s_Trial4_pipette temperature_thermistor.asc'
slope = 0.5


plt.rcParams.update({'font.size': 17})

df = pd.read_csv(path, delimiter = " ",skiprows=2,header=None)

V=df[0].values
pulse=df[1].values


t = [x / 10 for x in range(0, len(V))]
# t=numpy.arange(0, len(V), step)

wn =  np.arange(20000, 190000, dtype=int)
t_wn = np.array(t)[wn]
V_wn = np.array(V)[wn]
pulse_wn =np.array(pulse)[wn]


# print(p_off[0]-p_on[0])
# print(p_on, p_off)


fig, ax1 = plt.subplots()

ax1.plot(t_wn, V_wn)
ax1.set_title('laser and pipet')

ax2 = ax1.twinx()
ax2.plot(t_wn, pulse_wn, 'C1')
ax2.set_ylim([0, 20])

plt.savefig('pipette_pulse1.'+format, format=format)
#plt.show()

#p_df=np.diff(pulse_wn)
#%%
plt.figure()

p_on=np.where(np.diff(pulse)>2)
p_off=np.where(np.diff(pulse)<-2)
V_avg=np.empty([51500])
V_sm=np.empty( [51500])
for i in p_on[0][0:-1]:
    wn=np.arange(i-1000, i+50500)
    t_wn = np.array(t)[wn]
    pulse_wn =np.array(pulse)[wn]
    V_wn = np.array(V)[wn]
    V_ofs=np.mean(V_wn[:1000])
    V_wn=V_wn-V_ofs
    #V_sm=np.array([V_sm, V_wn], dtype=float)
    V_sm=np.vstack((V_sm, V_wn))
    #V_sm=np.append(V_sm, V_wn, axis=1)
    #np.concatenate(V_sm, V_wn)
    #V_sm=[V_sm, V_wn ]
    
    plt.plot(1000*V_wn)
    plt.xlabel("Time (time steps)")
    plt.ylabel("Vm (mV)")
    
    #print(wn)
#V_sm=np.asarray(V_wn)
plt.savefig('pipette_pulse2.'+format, format=format)
#print(np.mean(V_sm, axis=0, dtype=float))

print(1000*np.max(np.mean(V_sm, axis=0, dtype=float)[:10000]))
plt.figure()
plt.plot(t_wn/1000, 1000*np.mean(V_sm, axis=0, dtype=float))
plt.xlabel("Time (s)")
plt.ylabel("Vm (mV)")

# plt.show()
plt.savefig('pipette_pulse3.'+format, format=format)


V_sm *=slope

print(1000*np.max(np.mean(V_sm, axis=0, dtype=float)[:10000]))
plt.figure()
plt.plot(t_wn, 1000*np.mean(V_sm, axis=0, dtype=float))
plt.xlabel("Time (ms)")
plt.ylabel("Temperature (ºC)\nestimated with %.2f"%slope)
plt.tight_layout()
# plt.show()
plt.savefig('pipette_pulse_temp.'+format, format=format)

