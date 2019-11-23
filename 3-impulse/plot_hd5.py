import h5py
import matplotlib.pyplot as plt
import numpy as np
import sys

def plot_func(res, save):
	if save!='':
		np.savetxt(save, res, delimiter='')

	plt.plot(res)
	# plt.show()
	plt.close()


##File as argument
if len(sys.argv) > 2:
	path = sys.argv[1]
	file = sys.argv[2]
	filename = path + file
else:
	path = ''
	filename  = 'exp1.h5'

print(filename)


trial = "Trial1"

struct    = '/'+trial+'/Synchronous Data/Channel Data'
columna_a = 1 # 1 es impulsos, 2 es señal
columna_b = 2 # 1 es impulsos, 2 es señal
columna_c = 4 # 1 es impulsos, 2 es señal
columna_d = 5 # 1 es impulsos, 2 es señal

f = h5py.File(filename, 'r')
dset = f[struct]
data = dset.value
res1 = data[:,columna_a]
res2 = data[:,columna_b]
res3 = data[:,columna_c]
res4 = data[:,columna_d]

save     = path + file[:-3] + "_"+  trial +'_input_signal.asc'
# save     = ''
plot_func(res1, save)

save     = path + file[:-3] + "_"+  trial +'_living_signal.asc'
# save     = ''
plot_func(res2, save)

save     = path + file[:-3] + "_"+  trial +'_wang_nap.asc'
# save     = ''
plot_func(res3, save)

save     = path + file[:-3] + "_"+  trial +'_wang_ks.asc'
# save     = ''
plot_func(res4, save)

f.close()
