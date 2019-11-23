
import os,sys
from parse import *

if len(sys.argv) > 1:
	path = sys.argv[1]
	# file = sys.argv[2]
else:
	path = ''
	# file  = 'test-spikes.asc'

files = []

for f in os.listdir(path):
	if f.endswith(".asc"):
		files.append(f)

log_file = path +"global_stats.txt"
print(log_file)

for f in files:
	os.system("python3 spike_detection.py "+path+ " "+f)
	params = parse_params(f)
	labels = ' '.join(list(params.keys()))
	values = ' '.join(list(params.values()))
	# print()
	os.system("echo "+ labels +" >> "+log_file)
	os.system("echo "+ values +" >> "+log_file)
	os.system("echo Type Acc Dec >> "+log_file)
	os.system("python3 impulse.py "+ path + " " + f[:-4]+"/"+" >> "+log_file)
	os.system("echo end >> "+path +"global_stats.txt")
