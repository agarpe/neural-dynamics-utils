import h5py
import matplotlib.pyplot as plt
import numpy as np
import sys



##File as argument
if len(sys.argv) > 2:
	path = sys.argv[1]
	file = sys.argv[2]
	filename = path + file
else:
	path = ''
	filename  = 'exp1.h5'

print(filename)


for i in range(1,10):
	trial = "Trial"+str(i)

	struct    = '/'+trial+'/Synchronous Data/Channel Data'
	columna_a = 0 # impulsos, 2 es señal
	columna_e = 1 # synapse
	try:
		columna_b = 2 # señal
		columna_c = 4 # Wang Nap
		columna_d = 5 # Wang Ks
	except:
		pass

	try:
		f = h5py.File(filename, 'r')		
	except:
		break

	dset = f[struct]
	data = dset.value


	res = data[:,(columna_a,columna_b,columna_c,columna_d,columna_e)]
	name = filename[:-3]+"_"+trial+"_.asc"
	print(name,res.shape)
												# 'pre v wangNap wangKs synapse'
	np.savetxt(name, res, delimiter=' ',header='synapse v wangNap wangKs pulse')
