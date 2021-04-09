from matplotlib.lines import Line2D
from stats_plot_functions import *


plt.rcParams.update({'font.size': 30})



import argparse

# 	print("Example: python3 superpos_from_model.py ../../laser_model/HH/data/gna/ gna \"Gna simulation\" 0.001 8 20")
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=True, help="Path to the file to show stats from")
ap.add_argument("-m", "--mode", required=True, 
				help="Barchart plot mode: 'simple' for models and 'complete' for experimental. Write columns strings separated by space")
ap.add_argument("-dt", "--data_type", required=True, help="Whether data is obtain from models or experiments (control-laser-control)")
ap.add_argument("-c", "--cols", required=False,default=1, help="Number of columns in plot")
ap.add_argument("-sb", "--sort_by", required=False,default='time', help="Sort data by 'time' or 'name'")
ap.add_argument("-s", "--selection", required=False,default='n', help="Spike selection of no_bursts and burst spikes")
ap.add_argument("-pe", "--path_extension", required=False,default="", help="Subpath where pkl files are located. p.e. events")
ap.add_argument("-sa", "--save", required=False, default='n', help="Option to save plot file")
ap.add_argument("-sh", "--show", required=False, default='y', help="Option to show plot file")
ap.add_argument("-v", "--verbrose", required=False, default='n', help="Option to verbrose actions")
ap.add_argument("-log", "--log", required=False, default='n', help="Option to log best trial selection")
args = vars(ap.parse_args())


path = args['path']
plot_mode = args['mode'] 
data_type = args['data_type']

ext_path = args['path_extension'] #subpath where 
sort_by = args['sort_by']
show= True if args['show']=='y' else False 
save= True if args['save']=='y' else False 
log= True if args['log']=='y' else False 
verb= True if args['verbrose']=='y' else False 
spike_selection= True if args['selection']=='y' else False 
cols=int(args['cols'])

plot_diffs = True if plot_mode=='complete' else False

if data_type=='model' and plot_diffs:
	print("Error: model cannot be ploted with diffs (run in simple mode)")
	exit()


if data_type=='experimental':
	n_spikes_labels = ['control_pre_count','laser_count','control_pos_count']
	duration_labels = ['control_pre_duration','laser_duration','control_pos_duration']
	amplitude_labels = ['control_pre_amplitude','laser_amplitude','control_pos_amplitude']
	slope_dep_labels = ['control_pre_slope_dep','laser_slope_dep','control_pos_slope_dep']
	slope_rep_labels = ['control_pre_slope_rep','laser_slope_rep','control_pos_slope_rep']

	colors = ['cornflowerblue','indianred','royalblue']
	# colors = ['cornflowerblue','firebrick','olivedrab']
	legends = ['control_pre','laser','control_pos']

elif data_type=='model':
	n_spikes_labels = ['count']
	duration_labels = ['duration']
	amplitude_labels = ['amplitude']
	slope_dep_labels = ['slope_dep']
	slope_rep_labels = ['slope_rep']

	colors = ['b']

	legends = []

titles = {'spikes':{'labels':n_spikes_labels,'title':n_spike_title},'duration':{'labels':duration_labels,'title':duration_title},
'amplitude':{'labels':amplitude_labels,'title':amplitude_title},'slope_dep':{'labels':slope_dep_labels,'title':slope_dep_title},
'slope_rep':{'labels':slope_rep_labels,'title':slope_rep_title}}


if plot_mode=="complete" or plot_mode=="simple":
	columns = ['duration','amplitude','slope_rep','slope_dep','spikes']
	# columns = ['duration','amplitude','slope_rep','slope_dep']
else:
	columns = [plot_mode]


# plt.figure(figsize=(15,4*len(columns))) #best size for 4 rows
# cols =1
fig_setup=(cols,len(columns)//cols+len(columns)%cols)
# fig_setup=(2,2)
plt.figure(figsize=(20,8*fig_setup[1])) 


dirs = sorted(glob.glob(path+"*%s*"%""))
print(dirs)

if(dirs==[]):
	print("Error: No dirs found. Check the extension provided")
	exit()


labels=[]
ignored=0

if log:
	trial_log = open(path+"best_trial_id.log","w")
	trial_log.write("Experiment_day Trial_name Trial_number(startsfrom0) difference_value_pre(ms) difference_value_pos(ms)\n")
#Iterates over all directories in the general dir. given as argument. 
for i,d in enumerate(dirs):
	dir_name = d[d.rfind("/")+1:]

	#ignore regular files
	if dir_name.find(".")!=-1:
		ignored +=1
		continue

	if not spike_selection and dir_name.find("burst")!=-1:
		ignored +=1
		continue


	print(dir_name)

	all_trials=[] #reset one day trials list.
	# print(d+"/events/*.pkl")
	files = glob.glob(d+"/"+ext_path+"/*.pkl")
	if sort_by == 'time':
		files.sort(key=os.path.getmtime)
	elif sort_by == 'name':
		files.sort(key=os.path.basename)


	best_trial = (0,0)
	#Concat all trials from one same experiment day into one df and plots it.
	for j,f in enumerate(files):
		df = pd.read_pickle(f)
		# print(df.describe())
		# print(j,f)

		# # df["Trial"]=j # adds Trial reference to the data frame.
		# if data_type=='experimental':
		try:
			for n_l,d_l in zip(n_spikes_labels,duration_labels):
				df[n_l] = df[d_l].count()
		except: #in case labels are not found.
			for n_l,d_l in zip(n_spikes_labels,duration_labels):
				df[n_l] = 0

		#In models each pkl is a set of 50 spikes with a specific parameters config
		if data_type == 'model': 
			all_trials.append(df)
		else:
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
		if data_type == 'model':
			df_best=pd.concat(all_trials)
		try:
			if verb:
				print(df_best.describe())
			plot_barchart(df_best,i-ignored,labels,plot_diffs=plot_diffs,fig_size=fig_setup
				,columns=columns,colors=colors,titles=titles,legends=legends)
			if log:
				trial_log.write("%s %s %d %d %d\n"%(
					dir_name,best_trial_f,best_trial_id,best_trial[0],best_trial[1]))
		except Exception as e:
			print("failed %s"%dir_name)
			print(e)
			pass
	else:
		ignored +=1

print(labels)
if log:
	trial_log.close()

plt.tight_layout()

if spike_selection:
	plot_mode+="_selection"

if save:
	plt.savefig(path+"general_barchart_bests_"+plot_mode+".eps",format="eps")
	plt.savefig(path+"general_barchart_bests_"+plot_mode+".png",format="png")
if show:
	plt.show()
