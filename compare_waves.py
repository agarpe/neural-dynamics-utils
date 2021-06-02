import charact_utils as utils 
import argparse 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import scipy.signal as signal


ap = argparse.ArgumentParser()
ap.add_argument("-p1", "--path1", required=True, help="Path to the file to analyze")
# ap.add_argument("-p2", "--path2", required=True, help="Path to the file to analyze")
ap.add_argument("-s", "--show", required=False,default='y', help="Show plot")
ap.add_argument("-sv", "--save", required=False,default='y', help="Save events")
args = vars(ap.parse_args())


path = args['path1']
# path2 = args['path2']
show= True if args['show']=='y' else False 
save= True if args['save']=='y' else False 


try:
	df = pd.read_csv(path, delimiter = " ",skiprows=1,header=None)
	# df2 = pd.read_csv(path2, delimiter = " ",skiprows=1,header=None)
except:
	print("Error: file not found",path)
	exit()

time = df[0]
data1 = df[1] #left electrode
data2 = df[2] #right electrode 
time = np.arange(0,data1.shape[0],1)*0.1

plt.figure(figsize=(20,15))
plt.plot(time,data1,label='VD1')
plt.plot(time,data2,label='RPD2')
plt.legend()

if save:
	plt.savefig(path[:-4]+"_comparation.eps",format='eps')

ini = 1500000
end = 1600000
plt.figure(figsize=(20,15))
plt.plot(time[ini:end],data1[ini:end],label='VD1')
plt.plot(time[ini:end],data2[ini:end],label='RPD2')
plt.legend()

if save:
	plt.savefig(path[:-4]+"_comparation_zoom.eps",format='eps')

if show:
	plt.show()