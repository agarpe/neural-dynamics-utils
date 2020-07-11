########
#IMPORTS AND ARGUMENTS
########
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse
import time
import datetime

# Lee los archivos del RTBiomanager antiguo instalado en ceres
class read_data_ceres ():
	def __init__(self, filename, t_start, t_end, freq):

		header = (int(t_start)*freq) 
		if t_end!= 'None':
			nrows = (int(t_end)-int(t_start))*freq
			dataset = pd.read_csv(filename, delimiter=' ', header=header, nrows=nrows)
		else:
			dataset = pd.read_csv(filename, delimiter=' ', header=header)
		
		array = dataset.values
		data = array[:,:]

		# El tiempo no es absoluto
		self.t = data[:, 0] / 1000.0
		inicial = self.t[0]
		for i in range(len(self.t)):
			self.t[i] = self.t[i]-inicial

		self.v_m = data[:, 3]
		self.v_l = data[:, 2]
		self.c_on = False

		# Cuadrar señales
		for i in range(len(self.v_m)):
			self.v_m[i] = self.v_m[i] - 2.0

		self.num_points = len(self.t)

		self.v_min = min (min(self.v_l), min(self.v_m)) - .1
		self.v_max = max (max(self.v_l), max(self.v_m)) + .1

		self.range = self.v_max - self.v_min
		self.pos1  = self.v_min + (self.range*0.6)
		self.pos2  = self.v_min + (self.range*0.8)
		self.pos3  = self.v_min + (self.range*0.1)



# Lee la primera version de ficheros de Manu y Roy
# From: ¿? de 2017
# To  : ¿? de 2018
class read_data_manuroy_v1 ():
	def __init__(self, filename, t_start, t_end, freq):
		'''
		# Leer mas cosas del fichero
		file = open(filename,'r')
		line = file.readline()
		channels = line.split(' ')
		n_in_chan = int(channels[0])
		n_out_chan = int(channels[1]) 
		file.close()
		'''

		header = 1+(int(t_start)*freq) 

		if t_end!= 'None':
			nrows = (int(t_end)-int(t_start))*freq
			dataset = pd.read_csv(filename, delimiter=' ', header=header, nrows=nrows)
		else:
			dataset = pd.read_csv(filename, delimiter=' ', header=header)
		
		array = dataset.values
		data = array[:,:]

		#self.t_unix  = data[:, 0]
		self.t        = data[:, 1] / 1000.0 #Asi lo guardo en segundos
		#self.i       = data[:, 2]
		#self.lat     = data[:, 3]
		#v_model      = data[:, 4]
		self.v_m      = data[:, 5]
		self.c_on     = True
		self.c_m      = data[:, 6]
		self.c_l      = data[:, 7]
		self.v_l      = data[:, 8]

		'''
		# Leer mas cosas del fichero
		data_in = []
		for j in range(8, 8 + n_in_chan):
		    data_in.append(data[:, j])

		data_out = []
		for j in range(8 + n_in_chan, 8 + n_in_chan + n_out_chan):
		    data_out.append(data[:, j])
		'''

		### Datos útiles
		self.num_points = len(self.t)

		self.v_min = min (min(self.v_l), min(self.v_m)) - .1
		self.v_max = max (max(self.v_l), max(self.v_m)) + .1
		self.c_min = min (min(self.c_l), min(self.c_m))
		self.c_min = -0.5 # xD
		self.c_max = max (max(self.c_l), max(self.c_m))

		self.range = self.v_max - self.v_min
		self.pos1  = self.v_min + (self.range*0.6)
		self.pos2  = self.v_min + (self.range*0.8)
		self.pos3  = self.v_min + (self.range*0.1)


# Lee el fichero 'datarafi' producido por el script MATLAB
# El MATLAB a su vez necesita de los eventos extraidos en dataview
# Ahora tambien pueden ser extraidos mediante los plots de RTHybrid
class read_events_datarafi ():
	def __init__(self, filename, t_start, t_end, freq):
		dataset = pd.read_csv(filename, delimiter=' ', header=None) # Si header = 0, no lee la primera linea
		# dataset = pd.read_csv(filename, delimiter=',', header=None) # Si header = 0, no lee la primera linea
		array = dataset.values
		data = array[:,:]

		self.firstLP  = data[:, 1] / 1000.0 #Asi lo guardo en segundos
		self.lastLP   = data[:, 2] / 1000.0 #Asi lo guardo en segundos
		self.firstPD  = data[:, 3] / 1000.0 #Asi lo guardo en segundos
		self.lastPD   = data[:, 4] / 1000.0 #Asi lo guardo en segundos
		self.burstLP = data[:, 5]
		self.burstPD = data[:, 6]
		self.fPD_fPD = data[:, 7] #Period
		self.lLD_fPD = data[:, 8] #Lp-pd delay
		self.fLP_fPD = data[:, 9] #lp-pd interval

		if t_end == 'None':
			t_end = float('inf')
		else:
			t_end = float(t_end)
		t_start = float(t_start)

		index_start = 0
		index_end   = len(self.firstPD)
		for i in range(len(self.firstPD)):
			if t_start>self.firstPD[i]:
				index_start = i
			elif t_end<self.firstPD[i]:
				index_end = i+1
				break

		self.firstLP  = self.firstLP [index_start:index_end]
		self.lastLP   = self.lastLP  [index_start:index_end]
		self.firstPD  = self.firstPD [index_start:index_end]
	
		self.lastPD   = self.lastPD  [index_start:index_end]

		##########
		for i in range(len(self.firstLP)):
			self.firstLP[i] = self.firstLP[i] - float(t_start)
		for i in range(len(self.lastLP)):
			self.lastLP[i] = self.lastLP[i] - float(t_start)
		for i in range(len(self.firstPD)):
			self.firstPD[i] = self.firstPD[i] - float(t_start)
		for i in range(len(self.lastPD)):
			self.lastPD[i] = self.lastPD[i] - float(t_start)

		##########

		self.fPD_fPD  = self.fPD_fPD [index_start:index_end]
		self.lLD_fPD  = self.lLD_fPD [index_start:index_end]
		self.fLP_fPD  = self.fLP_fPD [index_start:index_end]

		self.num_events = len(self.fPD_fPD)

def get_section(interval,start, end):
	interval = interval[start:end]
	return interval

# Lee el fichero NX burst events de lymnaea
# Formato de archivo: fist_spk last_spk
class read_events_lymnaea ():
	def __init__(self, filename, t_start, t_end, freq):
		dataset = pd.read_csv(filename, delimiter=' ', header=0) # Si header = 0, no lee la primera linea
		array = dataset.values
		data = array[:,:]

		self.firstN1  = data[:, 1] / 1000.0 #Asi lo guardo en segundos
		self.lastN1   = data[:, 2] / 1000.0 #Asi lo guardo en segundos
		self.firstN2  = data[:, 3] / 1000.0 
		self.lastN2   = data[:, 4] / 1000.0 
		self.firstN3  = data[:, 5] / 1000.0 
		try:
			self.lastN3   = data[:, 6] / 1000.0 
		except:
			print(data[:,6])
		self.fN1_fN1 = data[:, 7]/ 1000.0
		self.burstN1 = data[:, 8] / 1000.0 
		self.burstN2 = data[:, 9] / 1000.0 
		self.burstN3 = data[:, 10] / 1000.0
		self.fN1_fN2 = data[:, 11]/ 1000.0
		self.lN1_fN2 = data[:, 12]/ 1000.0 

		self.fN2_fN1 = data[:, 13]/ 1000.0
		self.lN2_fN1 = data[:, 14]/ 1000.0##??? equivalente? anterior LD seguramente sea PD??

		self.fN1_fN3 = data[:, 15]/ 1000.0 #N1N3interval
		self.lN1_fN3 = data[:, 16]/ 1000.0 #N1N3delay

		self.fN3_fN1 = data[:, 17]/ 1000.0 #N3N1interval
		self.lN3_fN1 = data[:, 18]/ 1000.0 #N3N1delay

		self.fN2_fN3 = data[:, 19]/ 1000.0 #N2N3interval
		self.lN2_fN3 = data[:, 20]/ 1000.0 #N2N3delay

		self.fN3_fN2 = data[:, 21]/ 1000.0 #N3N2interval
		self.lN3_fN2 = data[:, 22]/ 1000.0 #N3N2delay

		if t_end == 'None':
			t_end = float('inf')
		else:
			t_end = float(t_end)
		t_start = float(t_start)

		index_start = 0
		index_end   = len(self.firstN1)
		for i in range(len(self.firstN1)):
			if t_start>self.firstN1[i]:
				index_start = i
			elif t_end<self.firstN1[i]:
				index_end = i+1
				break

		self.firstN2  = self.firstN2 [index_start:index_end]
		self.lastN2   = self.lastN2  [index_start:index_end]
		self.firstN1  = self.firstN1 [index_start:index_end]
		self.lastN1   = self.lastN1  [index_start:index_end]
		self.firstN3  = self.firstN3 [index_start:index_end]
		self.lastN3   = self.lastN3  [index_start:index_end]

		##########
		for i in range(len(self.firstN2)):
			self.firstN2[i] = self.firstN2[i] - float(t_start)
		for i in range(len(self.lastN2)):
			self.lastN2[i] = self.lastN2[i] - float(t_start)
		for i in range(len(self.firstN1)):
			self.firstN1[i] = self.firstN1[i] - float(t_start)
		for i in range(len(self.lastN1)):
			self.lastN1[i] = self.lastN1[i] - float(t_start)
		for i in range(len(self.firstN3)):
			self.firstN3[i] = self.firstN3[i] - float(t_start)
		for i in range(len(self.lastN3)):
			self.lastN3[i] = self.lastN3[i] - float(t_start)

		##########

		self.fN1_fN1  = self.fN1_fN1 [index_start:index_end]
		self.fN1_fN2 = self.fN1_fN2 [index_start:index_end]
		self.lN1_fN2 = self.lN1_fN2 [index_start:index_end]

		self.fN2_fN1 = self.fN2_fN1 [index_start:index_end]
		self.lN2_fN1 = self.lN2_fN1 [index_start:index_end]

		self.fN1_fN3 = self.fN1_fN3 [index_start:index_end]
		self.lN1_fN3 = self.lN1_fN3 [index_start:index_end]

		self.fN3_fN1 = self.fN3_fN1 [index_start:index_end]
		self.lN3_fN1 = self.lN3_fN1 [index_start:index_end]

		self.fN2_fN3 = self.fN2_fN3 [index_start:index_end]
		self.lN2_fN3 = self.lN2_fN3 [index_start:index_end]

		self.fN3_fN2 = self.fN3_fN2 [index_start:index_end]
		self.lN3_fN2 = self.lN3_fN2 [index_start:index_end]


		self.num_events = len(self.fN1_fN1)

# Lee los archivos del RTBiomanager antiguo instalado en ceres
class read_data_lymnaea ():
	def __init__(self, filename, t_start, t_end, freq,c=False):

		header = (int(t_start)*freq) 
		if t_end!= 'None':
			nrows = (int(t_end)-int(t_start))*freq
			dataset = pd.read_csv(filename, delimiter=' ', header=header, nrows=nrows)
		else:
			dataset = pd.read_csv(filename, delimiter=' ', header=header)
		
		array = dataset.values
		data = array[:,:]

		# El tiempo no es absoluto
		self.t = data[:, 0] / 1000.0
		inicial = self.t[0]
		for i in range(len(self.t)):
			self.t[i] = self.t[i]-inicial

		self.v_n1 = data[:, 2]
		self.v_n2 = data[:, 3]
		self.v_n3 = data[:, 4]
		
		self.c_on = c
		if c:
			self.c = data[:,5]
			self.c_max = max(self.c)
			self.c_min = min(self.c)

		# Cuadrar señales
		for i in range(len(self.v_n1)):
			self.v_n1[i] = self.v_n1[i] - 2.0

		self.num_points = len(self.t)

		self.v_min = min (min(self.v_n2), min(self.v_n1),min(self.v_n3)) - .1
		self.v_max = max (max(self.v_n2), max(self.v_n1),max(self.v_n3)) + .1

		self.range = self.v_max - self.v_min
		self.pos1  = self.v_min + (self.range*0.6)
		self.pos2  = self.v_min + (self.range*0.8)
		self.pos3  = self.v_min + (self.range*0.1)
		self.pos4  = self.v_min + (self.range*0.4)

