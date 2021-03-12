from matplotlib.lines import Line2D
from stats_plot_functions import *


plt.rcParams.update({'font.size': 17})


import argparse

# 	print("Example: python3 superpos_from_model.py ../../laser_model/HH/data/gna/ gna \"Gna simulation\" 0.001 8 20")
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=True, help="Path to the file to show stats from")
ap.add_argument("-m", "--mode", required=True, help="Barchart plot mode: 'simple' for models and 'complete' for experimental")
ap.add_argument("-s", "--selection", required=False,default='n', help="Spike selection of no_bursts and burst spikes")
ap.add_argument("-pe", "--path_extension", required=False,default="", help="Path extension to the files to show stats from")
ap.add_argument("-sa", "--save", required=False, default='y', help="Option to save plot file")
ap.add_argument("-sh", "--show", required=False, default='y', help="Option to show plot file")
args = vars(ap.parse_args())


path = args['path']
plot_mode = args['mode'] 
ext_path = args['path_extension'] #name of the parameter varied during simulations
show= True if args['show']=='y' else False 
save= True if args['save']=='y' else False 
spike_selection= True if args['selection']=='y' else False 


colors = ['cornflowerblue','indianred','royalblue']
# colors = ['cornflowerblue','firebrick','olivedrab']

dirs = sorted(glob.glob(path+"*%s*"%""))
# dirs.sort(key=os.path.getmtime)

print(dirs)

if(dirs==[]):
	print("Error: No dirs found. Check the extension provided")

if plot_mode=="complete":
	columns = ['duration','amplitude','slope_rep','slope_dep','spikes']
else:
	columns = [plot_mode]
plt.figure(figsize=(15,8*len(columns)))

labels=[]
ignored=0

trial_log = open(path+"best_trial_id.log","w")
trial_log.write("Experiment_day Trial_name Trial_number(startsfrom0) difference_value_pre(ms) difference_value_pos(ms)\n")
#Iterates over all directories in the general dir. given as argument. 
for i,d in enumerate(dirs):
	dir_name = d[d.rfind("/")+1:]
	print(dir_name)

	#ignore regular files
	if dir_name.find(".")!=-1:
		ignored +=1
		continue

	if not spike_selection and dir_name.find("burst")!=-1:
		ignored +=1
		continue

	all_trials=[] #reset one day trials list.
	# print(d+"/events/*.pkl")
	files = glob.glob(d+"/"+ext_path+"/*.pkl")
	files.sort(key=os.path.getmtime)

	best_trial = (0,0)
	#Concat all trials from one same experiment day into one df and plots it.
	for j,f in enumerate(files):
		df = pd.read_pickle(f)
		# print(df.describe())
		# print(j,f)

		# df["Trial"]=j # adds Trial reference to the data frame.
		try:
			df["control_pre_count"]=df["control_pre_duration"].count() # adds each trial number of spikes count
			df["laser_count"]=df["laser_duration"].count() # adds each trial number of spikes count
			df["control_pos_count"]=df["control_pos_duration"].count() # adds each trial number of spikes count
		except:
			df["control_pre_count"]=0 # adds each trial number of spikes count
			df["laser_count"]=0 # adds each trial number of spikes count
			df["control_pos_count"]=0 # adds each trial number of spikes count

		try:
			dur_means = df[duration_labels].mean()
			laser_diff_pre,laser_diff_pos=get_diffs(dur_means)[1:] #Get duration mean in this Trial
			if(laser_diff_pre>best_trial[0] and laser_diff_pos>best_trial[1] ):
				trial_id = int(f[f.find("exp")+3])
				df["Trial"]=trial_id
				df_best = df 
				best_trial = (laser_diff_pre,laser_diff_pos)
				best_trial_id = trial_id
				best_trial_f=f[f.find("exp"):]
				# print(j)
		except Exception as e:
			print("Skiping",f)
			print(e)
			pass

	#Print experiment stats:
	if len(files) >0: #If no trials on directory --> ignore data.
		labels.append(dir_name) #Add label to list.
		try:
			# all_trials=pd.concat(all_trials)
			# print(df_best.describe())
			plot_barchart(df_best,i-ignored,labels,plot_diffs=False,cols=1,columns=columns,colors=colors)
			trial_log.write("%s %s %d %d %d\n"%(dir_name,best_trial_f,best_trial_id,best_trial[0],best_trial[1]))
		except:
			print("failed %s"%dir_name)
			pass
	else:
		ignored +=1

print(labels)
trial_log.close()

plt.tight_layout()

if spike_selection:
	plot_mode+="_selection"

if save:
	plt.savefig(path+"general_barchart_bests_"+plot_mode+".eps",format="eps")
	plt.savefig(path+"general_barchart_bests_"+plot_mode+".png",format="png")
if show:
	plt.show()
