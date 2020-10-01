import h5py
import matplotlib.pyplot as plt
import numpy as np
import sys

#################################################################################################
###	Parse h5 file to plain text, each Trial is save in a different file.
#################################################################################################


##File as argument
if len(sys.argv) > 2:
	path = sys.argv[1]
	file = sys.argv[2]
	filename = path + file
else:
	path = ''
	filename  = 'exp1.h5'

print(filename)


headers = 'pre synapse v vout wangNap wangKs'
##!!!!!! may vary depending on the recording
columns = (1,0,2,3,4,5) #Corresponding value in HDF5 group!!

#Open file 
try:
	f = h5py.File(filename, 'r')		
except:
	print("File not valid")
	exit()

for i in range(1,10):
	trial = "Trial"+str(i)

	struct    = '/'+trial+'/Synchronous Data/Channel Data'

	try:
		dset = f[struct]
		data = dset.value
	except:
		print("No trials left")
		break

	res = data[:,columns]
	name = filename[:-3]+"_"+trial+"_.asc"
	print(name,res.shape)
												
	np.savetxt(name, res, delimiter=' ',header=headers)
