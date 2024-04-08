import numpy as np
import pandas as pd
import pickle as pkl
import sys

try:
	name = sys.argv[1]
	data = pd.read_csv(name, delimiter = " ",skiprows=2,header=None)
	data = data.values
except Exception as e:
	print("Error reading file:",e)
	exit()

try:
	pkl_name = name[:-4] + '.pkl'
	with open(pkl_name,'wb') as f:
		pkl.dump(data, f)
except Exception as e:
	print("Error during parsing:",e)
	exit()

print("File %s created"%pkl_name)


print("Testing file just created:",pkl_name)
with open(pkl_name,'rb') as f2:
	data2 = pkl.load(f2)
print(data2.shape)
# print(waveforms)