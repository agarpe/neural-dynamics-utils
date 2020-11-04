import pandas as pd 
import sys
import glob
import os
import matplotlib.pyplot as plt

if len(sys.argv) ==3:
	path = sys.argv[1]
	extension = sys.argv[2]
elif len(sys.argv) ==2:
	path = sys.argv[1]
	extension = ""
else:
	print("Use: python3 autom_spike_detection.py path [extension]")
	exit()

files = glob.glob(path+"*%s*.asc"%extension)
files.sort(key=os.path.getmtime)


all_trials=[]
print(files)

if(files==[]):
	print("Error: No files found. Check the extension provided")

for i,f in enumerate(files):
	print(f)
	os.system("python3 get_spike_events.py -s False -p "+f)