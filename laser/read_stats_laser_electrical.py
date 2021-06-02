
import argparse 
import glob
import os
import pandas as pd

ap = argparse.ArgumentParser()

ap.add_argument("-p", "--path", required=True, help="Path to the file to analyze")
# ap.add_argument("-s", "--show", required=False,default=True, help="Show plot")
# ap.add_argument("-sv", "--save", required=False,default=True, help="Save events")
args = vars(ap.parse_args())


path = args['path']

files = glob.glob(path+"*.pkl")
files.sort(key=os.path.getmtime)


all_trials=[]
print(files)

if(files==[]):
	print("Error: No files found. Check the extension provided")
	exit()

for i,f in enumerate(files):
	print(f)
	
	
	# df = pd.read_pickle(f)
	# print(df.describe())