from matplotlib.lines import Line2D
from stats_plot_functions import *


plt.rcParams.update({'font.size': 17})


if len(sys.argv) ==3:
	path = sys.argv[1]
	extension = sys.argv[2]
elif len(sys.argv) ==2:
	path = sys.argv[1]
	extension = ""
else:
	print("Use: python3 stats_plot.py path")
	exit()


plot_type = "simple"
ext_path = ""
# ext_path = "events"



dirs = sorted(glob.glob(path+"*%s*"%extension))
# dirs.sort(key=os.path.getmtime)

print(dirs)

if(dirs==[]):
	print("Error: No dirs found. Check the extension provided")

plt.figure(figsize=(15,20))

labels=[]
ignored=0
#Iterates over all directories in the general dir. given as argument. 
for i,d in enumerate(dirs):
	dir_name = d[d.rfind("/")+1:]
	print(dir_name)

	#ignore regular files
	if dir_name.find(".")!=-1:
		continue

	all_trials=[] #reset one day trials list.
	# print(d+"/events/*.pkl")
	files = glob.glob(d+"/"+ext_path+"/*.pkl")
	files.sort(key=os.path.getmtime)

	#Concat all trials from one same experiment day into one df and plots it.
	for j,f in enumerate(files):
		df = pd.read_pickle(f)
		# print(df.describe())

		df["Trial"]=j # adds Trial reference to the data frame.
		all_trials.append(df) #appends df to all trials list

	if len(files) >0: #If no trials on directory --> ignore data.
		labels.append(dir_name) #Add label to list.
		try:
			all_trials=pd.concat(all_trials)
			if plot_type=="complete":
				plot_barchart(all_trials,i-ignored,labels)
			elif plot_type=="simple":
				plot_barchart_simple(all_trials,i-ignored,labels)
				path2 = path+ext_path+"/"+dir_name+"/"
				os.system("python3 stats_plot_model.py "+path2)
		except:
			pass
	else:
		ignored +=1

print(labels)


plt.tight_layout()

# plt.savefig(path+"general_barchart_"+plot_type+".eps",format="eps")
# plt.show()
