from matplotlib.lines import Line2D
from stats_plot_functions import *


plt.rcParams.update({'font.size': 17})


import argparse

# 	print("Example: python3 superpos_from_model.py ../../laser_model/HH/data/gna/ gna \"Gna simulation\" 0.001 8 20")
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=True, help="Path to the file to show stats from")
ap.add_argument("-m", "--mode", required=True, help="Barchart plot mode: 'simple' for models and 'complete' for experimental")
ap.add_argument("-pe", "--path_extension", required=False,default="", help="Path extension to the files to show stats from")
ap.add_argument("-m", "--mode", required=True, help="Path to the file to show stats from")
ap.add_argument("-sa", "--save", required=False, default='y', help="Option to save plot file")
ap.add_argument("-sh", "--show", required=False, default='y', help="Option to show plot file")
args = vars(ap.parse_args())


path = args['path']
plot_mode = args['mode'] 
extension = args['path_extension'] #name of the parameter varied during simulations
show= True if args['show']=='y' else False 
save= True if args['save']=='y' else False 





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
			if plot_mode=="complete":
				plot_barchart(all_trials,i-ignored,labels)
			elif plot_mode=="simple":
				plot_barchart_simple(all_trials,i-ignored,labels)
				path2 = path+ext_path+"/"+dir_name+"/"
				os.system("python3 stats_plot_model.py -p"+path2)
		except:
			pass
	else:
		ignored +=1

print(labels)


plt.tight_layout()

# plt.savefig(path+"general_barchart_"+plot_mode+".eps",format="eps")
# plt.show()
