import os
import glob
import sys


path = sys.argv[1]

paths = glob.glob(path)

print(paths)

for path in paths:
	print(path)
	cmd = "python3 ~/Workspace/scripts/pkl_parser.py %s"%path
	os.system(cmd)
	print(cmd)