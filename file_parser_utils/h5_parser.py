import numpy as np
import pandas as pd
import h5py
import sys

try:
	name = sys.argv[1]
	data = pd.read_csv(name, delimiter = " ",skiprows=2,header=None)
	data = data.values
except Exception as e:
	print("Error reading file:",e)
	exit()

try:
	h5_name = name[:-4]+'.pkl'
	with h5py.File(h5_name, "w") as f:
		dset = f.create_dataset('dataset', data=data)

except Exception as e:
	print("Error during parsing:",e)
	exit()

print("File %s created"%h5_name)

print("Testing file just created:",h5_name)

h5f = h5py.File(h5_name,'r')
data = h5f['dataset'][:]
h5f.close()

print(data.shape)
print(data)
