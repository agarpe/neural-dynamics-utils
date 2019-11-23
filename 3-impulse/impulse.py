#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 16})

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

import charact_utils as utils
# from charact_utils import *


# ## Define Path

# In[2]:


folder = "/home/alicia/Documentos/data/3-impulse/14-11-2019/"
exp_ = "exp3/"


if(len(sys.argv)>2):
    folder = sys.argv[1]
    exp_ = sys.argv[2]

elif(len(sys.argv)>1):
    exp_ = sys.argv[1]
    folder = ''

path = folder+exp_
FIGSIZE = (24,6)
acc_color = 'forestgreen'
dec_color = 'firebrick'

# print(path)
# ## Carga archivos en secs

# In[3]:

#From DataView
# pulses = utils.read_spike_events(path + "events.txt",dt=0.001)
# spikes = utils.read_spike_events(path + "spikes.txt",dt=0.001)


#From Script
pulses = utils.read_spike_events(path + "events.txt",dataview=False,dt=0.1)
spikes = utils.read_spike_events(path + "spikes.txt",dataview=False,dt=0.1)


# In[4]:


#Getting 3 pulses events
first_p = pulses[0::3]
second_p = pulses[1::3]
third_p = pulses[2::3]


# ## Detectar spike en isis

# In[5]:


#Input: Pulse events 1, pulse events 2, spikes
#Output: array of index of events where there is a spike.
def check_spike(a,b,spikes):
    sp_indexes = []
    for i in range(a.shape[0]-1):
        #Guarda indices donde se de spike.
        elems = spikes[np.where((spikes > a[i]) & (spikes < b[i]))]
#         print(a[i],b[i],elems)
        if len(elems) >0 :
#             plt.plot(isi1[i],isi2[i],'.',color='blue')
            sp_indexes.append(i)
        
    return sp_indexes


# In[6]:


def plot_isi_cond(isi1,isi2,indexes,title,color='blue',label='',acc_decc=True):
    lim = max(max(isi1),max(isi2))+0.5
    #accelerating
    try: 
        isi1 = isi1[indexes]
        isi2 = isi2[indexes]

        if(acc_decc):
            x_1 = isi1[np.where(isi1>=isi2)]
            y_1 = isi2[np.where(isi1>=isi2)]
            
            x_2 = isi1[np.where(isi1<isi2)]
            y_2 = isi2[np.where(isi1<isi2)]
            
            scatter =  plt.scatter(x_1,y_1,c=acc_color,label="Accelerating")
            scatter =  plt.scatter(x_2,y_2,c=dec_color,label="Deccelerating")
            plt.legend()
        else:
    #         col = np.full(isi1.shape,color)
            scatter =  plt.scatter(isi1,isi2,c=color,label=label)
            

            
        plt.xlabel('ISI1 (s)')
        plt.ylabel('ISI2 (s)')
        plt.xlim(0,lim)
        plt.ylim(0,lim)
        plt.title(title)
        if(label!=''):
            plt.legend()
    except:
        pass


# ## Spikes según pulsos

# In[7]:


isi1 = second_p-first_p
isi2 = third_p-second_p
ibi = first_p[1:] -third_p[:-1]
# print(third_p[0],first_p[1])

indexes_1 = check_spike(first_p,second_p,spikes)
indexes_2 = check_spike(second_p,third_p,spikes)
indexes_3 = check_spike(third_p[:-1],first_p[1:],spikes)
indexes_3= np.array(indexes_3)


# In[8]:


fig = plt.figure(figsize=FIGSIZE)
plt.subplot(1,3,1)

plot_isi_cond(isi1,isi2,indexes_1,"Spike after 1st")

plt.subplot(1,3,2)

plot_isi_cond(isi1,isi2,indexes_2,"Spike after 2nd",'green')

plt.subplot(1,3,3)

plot_isi_cond(isi1,isi2,indexes_3,"Spike after 3rd",'red')


plt.savefig(path+"spikes_per_pulse.png")

# plt.show()
plt.close(fig)
# plt.plot(isi1,isi2,'.')


# In[9]:


ones = np.ones(pulses.shape)
ones_sp = np.ones(spikes.shape)*1.5
ones_index1 = np.zeros(first_p.shape)
ones_index1[indexes_1] = 1
ones_index2 = np.zeros(second_p.shape)
ones_index2[indexes_2] = 1
ones_index3 = np.zeros(third_p.shape)
ones_index3[indexes_3] = 1


plt.figure(figsize=(15,5))
plt.title("Eventos detectados de spikes en cada pulso")
# plt.xlim(0,100)
plt.plot(pulses,ones,'.',label="pulsos")
plt.plot(spikes,ones_sp,'x',label="spikes")
plt.plot(first_p,ones_index1,'s',label="Hay spike 1")
plt.plot(second_p,ones_index2,'s',label="Hay spike 2")
plt.plot(third_p,ones_index3,'s',label="Hay spike 3")
plt.legend()
# plt.show()


# ## Spikes condicionados por otros

# In[10]:


indexes_12 = np.intersect1d(indexes_1,indexes_2)
indexes_23 = np.intersect1d(indexes_2,indexes_3)
indexes_13 = np.intersect1d(indexes_1,indexes_3)


# In[11]:


fig= plt.figure(figsize=FIGSIZE)
plt.subplot(1,3,1)

plot_isi_cond(isi1,isi2,indexes_12,"Spikes after 1st & 2nd")
plt.subplot(1,3,2)
plot_isi_cond(isi1,isi2,indexes_23,"Spikes after 2nd & 3rd",'green')
plt.subplot(1,3,3)
plot_isi_cond(isi1,isi2,indexes_13,"Spikes after 1st & 3rd",'red')

plt.savefig(path+"conditioned_spikes.png")
# plt.show()
# plt.plot(isi1,isi2,'.')
plt.close(fig)

# ## Spikes únicos

# In[12]:


fig = plt.figure(figsize=FIGSIZE)
plt.subplot(1,3,1)

indexes_1_2 = np.setdiff1d(indexes_1,indexes_2)
indexes_1_uniq = np.setdiff1d(indexes_1_2,indexes_3)
plot_isi_cond(isi1,isi2,indexes_1_uniq,"Only spikes after 1st")

plt.subplot(1,3,2)

indexes_2_1 = np.setdiff1d(indexes_2,indexes_1)
indexes_2_uniq = np.setdiff1d(indexes_2_1,indexes_3)
plot_isi_cond(isi1,isi2,indexes_2_uniq,"Only spikes after pulse 2nd",'green')

plt.subplot(1,3,3)

indexes_3_1 = np.setdiff1d(indexes_3,indexes_1)
indexes_3_uniq = np.setdiff1d(indexes_3_1,indexes_2)
plot_isi_cond(isi1,isi2,indexes_3_uniq,"Only spikes after pulse 3rd",'red')
plt.savefig(path+"unique_spikes.png")
# plt.show()
# plt.plot(isi1,isi2,'.')
plt.close(fig)

# print("Ocurrencia eventos:")
# print("\t Solo Spike 1  ",len(indexes_1_uniq))
# print("\t Solo Spike 2  ",len(indexes_2_uniq))
# print("\t Solo Spike 3  ",len(indexes_3_uniq))
# print("\t After Spike 1  ",len(indexes_1_uniq))
# print("\t After Spike 2  ",len(indexes_2_uniq))
# print("\t After Spike 3  ",len(indexes_3_uniq))


# ## Activation to 2 & 3

# In[13]:



plt.figure(figsize=FIGSIZE)
plt.subplot(1,3,1)

plot_isi_cond(isi1,isi2,indexes_2,"Activation to 2nd")

plt.subplot(1,3,2)

plot_isi_cond(isi1,isi2,indexes_3,"Activation to 3rd",'green')

plt.subplot(1,3,3)

plot_isi_cond(isi1,isi2,indexes_23,"Activation to 2nd & 3rd",'red')


plt.savefig(path+"spikes_2_3_23.png")
plt.close(fig)
# plt.show()
# plt.plot(isi1,isi2,'.')


# In[ ]:


fig = plt.figure(figsize=(16,6))

plt.subplot(1,2,1)
# plot_isi_cond(isi1,isi2,indexes_3,"",label="3rd",acc_decc=False)
plot_isi_cond(isi1,isi2,indexes_3_uniq,"",color='cornflowerblue',label="only 3rd",acc_decc=False)
plot_isi_cond(isi1,isi2,indexes_23,"Only 3rd vs. 2nd y 3rd",color='navy',label="2 y 3",acc_decc=False)
plt.subplot(1,2,2)
# plot_isi_cond(isi1,isi2,indexes_2,"",label="2nd",acc_decc=False)
plot_isi_cond(isi1,isi2,indexes_2_uniq,"",color='cornflowerblue',label="only 2nd",acc_decc=False)
plot_isi_cond(isi1,isi2,indexes_23,"Only 2nd vs. 2nd y 3rd",color='navy',label="2 y 3",acc_decc=False)


plt.savefig(path+"comparative_2_3.png")
plt.close(fig)


# In[ ]:


def get_acc_dec(elems,i1,i2):
    elems = np.array(elems)
    acc = elems[np.where(i1[elems]>=i2[elems])]
    dec = elems[np.where(i1[elems]<i2[elems])]
    return acc,dec


def plot_stats(elems,title,labels,colors,print_=False):
    plt.title(title)
    x = range(len(elems))
    y = [len(elem) for elem in elems]
    plt.bar(x,y,width=0.1,label=labels,color=colors)
    plt.xticks(x, (labels))
    if print_:
        print(title.replace(" ","_"),len(elems[0]),len(elems[1]))


fig=plt.figure()
plot_stats([indexes_1,indexes_2,indexes_3],"Spikes after each pulse",['1st','2nd','3rd'],colors=['yellowgreen','yellowgreen','yellowgreen'])

plt.savefig(path+"stat_spikes.png")
plt.close(fig)

try:
    ad1 = get_acc_dec(indexes_1,isi1,isi2)
    fig=plt.figure()
    plot_stats(ad1,"Spikes after 1nd",["Acc.","Dec."],[acc_color,dec_color],print_=True)
    plt.savefig(path+"stat_1.png")
    plt.close(fig)
except:
    pass

try:
    ad2 = get_acc_dec(indexes_2,isi1,isi2)
    fig=plt.figure()
    plot_stats(ad2,"Spikes after 2nd",["Acc.","Dec."],[acc_color,dec_color],print_=True)
    plt.savefig(path+"stat_2.png")
    plt.close(fig)
except:
    pass

try:
    ad3 = get_acc_dec(indexes_3,isi1,isi2)
    fig=plt.figure()
    plot_stats(ad3,"Spikes after 3nd",["Acc.","Dec."],[acc_color,dec_color],print_=True)
    plt.savefig(path+"stat_3.png")
    plt.close(fig)
except:
    pass