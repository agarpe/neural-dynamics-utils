import charact_utils as utils 
import argparse 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import scipy.signal as signal


ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=True, help="Path to the file to analyze")
# ap.add_argument("-e", "--extension", required=True, help="File extension")
ap.add_argument("-c_neu", "--neucolumn", required=False,default=1, help="Column for neuron")
ap.add_argument("-c_curr", "--currcolumn", required=False,default=2, help="Column for current")
ap.add_argument("-sh", "--show", required=False,default='y', help="Show plot")
ap.add_argument("-sv", "--save", required=False,default='y', help="Save events")

args = vars(ap.parse_args())

path = args['path']

show= True if args['show']=='y' else False 
save= True if args['save']=='y' else False 

denoise =True


c_neu = int(args['neucolumn'])
c_curr = int(args['currcolumn'])

stim_th = -0.3 # when curr is below this is considered to be < 0
# ext = args['extension']

	
try:
	#WARNING!!!! skiprows 2 in case header is 2 lines. 
	df = pd.read_csv(path, delimiter = " ",skiprows=2,header=None)
except:
	print("Error: file not found",path)
	exit()


neu = df[c_neu]*10
curr = df[c_curr]/10
time = np.arange(0,neu.shape[0],1)*0.1

points = np.where(curr<stim_th)[0]

plt.subplot(2,1,1)
plt.plot(time,neu)
plt.plot(time[points],np.zeros(len(points)),'.')
plt.subplot(2,1,2)
plt.plot(time,curr)
plt.plot(time[points],np.zeros(len(points)),'.')
# plt.show()

change_curr =[]
change_neu =[]
aux_curr =[]
aux_neu =[]

off = True

for n1,n2,c1,c2 in zip(neu,neu[1:],curr,curr[1:]):
	if off: #stimulus starts
		if(c2 < stim_th):
			c0 = c1
			n0 = n1
			off = False
	else: #stimulus is on
		if(c2 >= stim_th): #end of stimulus
			off = True
			change_curr.append(min(aux_curr)-n0)
			change_neu.append(n0-min(aux_neu))
			print(min(aux_curr),c0)
			print(min(aux_neu),n0)

			aux_curr =[]
			aux_neu =[]
		else: #current injection on
			aux_curr.append(c1)
			aux_neu.append(n1)


print(len(change_neu),len(change_curr))
plt.figure(figsize=(10,10))

plt.plot(change_curr,change_neu,'.')
plt.xlabel("Current (nA)")
plt.ylabel("Neuron (mV)")
plt.show()



