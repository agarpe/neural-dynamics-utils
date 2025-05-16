import os
import glob
import sys


i_path = sys.argv[1]

print(i_path)
paths = glob.glob(i_path+'/*')

print(paths)

for path in paths:
	print(path)
	# if 'info' in path or 'pkl':
	# 	continue

	if '.asc' in path: 
		print(path)
		cmd = "python3 ~/Workspace/scripts/pkl_parser.py %s"%path
		print(cmd)
		os.system(cmd)
		# break
