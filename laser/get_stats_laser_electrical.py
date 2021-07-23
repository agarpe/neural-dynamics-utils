from superpos_functions import *
import argparse 
import glob
import os
import pandas as pd

def read_file(path,dt,name,logs,logs_names):
	try:
		events = read_from_events(path,max_cols=300,dt=0.1,dataview=True)
	except:
		print("Error: file 1 not found")
		exit()

	events = events.values

	log={}
	logs_names=[]
	logs=[]
	error_count=0
	for spike_i in range(events.shape[0]):
		#remove possible nan values:
		spike = events[spike_i,:][~np.isnan(events[spike_i,:])]
		get_spike_info(log,spike,dt,False,spike_i,error_count)

	if error_count>0:
		print("Spikes ignored:",error_count)

	logs_names.append('')
	logs.append(log)

	df = create_dataframe(logs,logs_names)
	# print(df.describe())
	init_path = path[:path.rfind("Exp")+4]+"_"
	os.system("mkdir -p "+init_path+name)
	df.to_pickle(init_path+name+"/info.pkl")


ap = argparse.ArgumentParser()

ap.add_argument("-p", "--path", required=True, help="Path to the file to analyze")
ap.add_argument("-na", "--names", required=True, help="All names in \"\" separated by white space")
ap.add_argument("-neu", "--neurons", required=True, help="Neuron names in \"\" separated by white space")

# ap.add_argument("-s", "--show", required=False,default=True, help="Show plot")
# ap.add_argument("-sv", "--save", required=False,default=True, help="Save events")
args = vars(ap.parse_args())


path = args['path']

names = args['names'].split()
neus = args['neurons'].split()



logs_names =[]
logs = []

for name in names:
	for neu in neus:
		file_path_n1=(path+"_"+name+"_"+neu+"_Waveform.txt")
		read_file(file_path_n1,0.1,name+"_"+neu+"_",logs,logs_names)

	# file_path_n2=(path+"_"+name+"_VD1"+"_Waveform.txt")
	# read_file(file_path_n2,0.1,name+"_VD1_",logs,logs_names)

