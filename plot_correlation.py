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
ap.add_argument("-scx", "--scale_x", required=False, default=1, help="Scale for Volts. Signal*scale")
ap.add_argument("-scy", "--scale_y", required=False, default=1, help="Scale for Volts. Signal*scale")
ap.add_argument("-sh", "--show", required=False,default='n', help="Show plot")
ap.add_argument("-sv", "--save", required=False,default='y', help="Save events")
# ap.add_argument("-sf", "--save_format", required=False,default='png', help="Save format")
ap.add_argument("-sf", "--save_format", required=False,default='pdf', help="Save format")
args = vars(ap.parse_args())


path = args['path']
cols = [int(c) for c in args['cols'].split()]
labels = args['labels'].split()
show= True if args['show']=='y' else False 
save= True if args['save']=='y' else False 
img_format = args['save_format']
scale_x = int(args['scale_x'])
scale_y = int(args['scale_y'])


try:
	df = pd.read_csv(path, delimiter = " ",skiprows=2,header=None)
	# df2 = pd.read_csv(path2, delimiter = " ",skiprows=1,header=None)
except:
	print("Error: file not found",path)
	exit()

# time = df[0]
data1 = df[cols[0]].values
data2 = df[cols[1]].values
label1 = labels[0]
label2 = labels[1]


x = data1*scale_x
y = data2*scale_y

x = x[np.where(y > 28.2)]
y = y[np.where(y > 28.2)]


r_sq,Y_pred,slope = utils.do_regression(x,y,'temp',False)
print("R2:", r_sq)
print("Slope:", slope)

with open(path[:-4] + '.txt', 'w') as f:
    f.write("R2: "+str(r_sq)+ "\n")
    f.write("Slope: "+str(slope)+ "\n")


plt.rcParams.update({'font.size': 17})

plt.scatter(x, y)
plt.xlabel(label1)
plt.ylabel(label2)
plt.title("R2 %.3f\n Slope %.3f"%(r_sq,slope))
plt.tight_layout()

if save:
	plt.savefig(path[:-4]+"_correlation."+img_format,format=img_format)


if show:
	plt.show()

# utils.plot_corr(data2, data1, 'temp', 'V',False,False)