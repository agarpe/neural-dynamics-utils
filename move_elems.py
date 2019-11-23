
import os,sys

if len(sys.argv) > 2:
	path1 = sys.argv[1]
	path2 = sys.argv[2]
	# file = sys.argv[2]
else:
	path = ''
	# file  = 'test-spikes.asc'

dest = path2 + "selection/"
os.system("mkdir -p "+dest)

for f in os.listdir(path1):
	if f.endswith(".png"):
		cmd = "cp "+path2+f[:-3]+"asc "+dest+f[:-3]+"asc"
		print(cmd)
		os.system(cmd)
