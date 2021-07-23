import charact_utils as utils 
import argparse 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import scipy.signal as signal

def plot(time,data1,data2):

	plt.figure(figsize=(20,15))
	plt.plot(time,data1,label=label1)
	plt.plot(time,data2,label=label2)
	plt.legend()

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=True, help="Path to the file to analyze")
ap.add_argument("-c", "--cols", required=True, help="Columns of waves")
ap.add_argument("-l", "--labels", required=True, help="Columns of waves")
ap.add_argument("-sc", "--scale", required=False, default=1, help="Scale for Volts. Signal*scale")
ap.add_argument("-sh", "--show", required=False,default='y', help="Show plot")
ap.add_argument("-sv", "--save", required=False,default='y', help="Save events")
ap.add_argument("-sf", "--save_format", required=False,default='png', help="Save format")
args = vars(ap.parse_args())


path = args['path']
cols = [int(c) for c in args['cols'].split()]
labels = args['labels'].split()
show= True if args['show']=='y' else False 
save= True if args['save']=='y' else False 
img_format = args['save_format']
scale = int(args['scale'])


try:
	df = pd.read_csv(path, delimiter = " ",skiprows=2,header=None)
	# df2 = pd.read_csv(path2, delimiter = " ",skiprows=1,header=None)
except:
	print("Error: file not found",path)
	exit()

# time = df[0]
data1 = df[cols[0]] * scale
data2 = df[cols[1]] * scale
label1 = labels[0]
label2 = labels[1]
time = np.arange(0,data1.shape[0],1)*0.1

plot(time,data1,data2)
if save:
	plt.savefig(path[:-4]+"_comparation."+img_format,format=img_format)

ini = 1500000
end = 1600000
plot(time[ini:end],data1[ini:end],data2[ini:end])

if save:
	plt.savefig(path[:-4]+"_comparation_zoom."+img_format,format=img_format)

if show:
	plt.show()