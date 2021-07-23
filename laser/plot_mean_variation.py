from matplotlib.lines import Line2D
from stats_plot_functions import *


plt.rcParams.update({'font.size': 40})

def plot_boxplot(df,columns,colors,legend,labels,labels_indexes,ylabel):
	bp = df.boxplot(return_type='dict',patch_artist=True,grid=False,column=columns,showfliers=False)

	colors = colors*4

	used =[]
	legend_patchs=[]

	for i,(patch,color) in enumerate(zip(bp['boxes'],colors)):
			patch.set_facecolor(color)
			if(color not in used):
				used.append(color)
				legend_patchs.append(patch)

	for patch in bp['medians']:
		plt.setp(patch, color='black',linewidth=1.5)

	plt.tight_layout()
	plt.xticks(labels_indexes, labels)
	plt.legend(legend_patchs,legend,loc='upper left')
	plt.ylabel(ylabel)



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
ap.add_argument("-se", "--save_extension", required=False, default='', help="Extension to path where file is saved")
ap.add_argument("-sa", "--save", required=False, default='n', help="Option to save plot file")
ap.add_argument("-sh", "--show", required=False, default='y', help="Option to show plot file")
ap.add_argument("-v", "--verbrose", required=False, default='n', help="Option to verbrose actions")
ap.add_argument("-log", "--log", required=False, default='n', help="Option to log best trial selection")
args = vars(ap.parse_args())


path = args['path']
plot_mode = args['mode'] 
data_type = args['data_type']

ext_path = args['path_extension'] #subpath where 
save_extension = args['save_extension'] #subpath where 
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

	# colors = ['cornflowerblue','indianred','royalblue']
	colors = ['cornflowerblue','firebrick','olivedrab']
	legends = ['control_pre','laser','control_pos']

elif data_type=='model':
	n_spikes_labels = ['count']
	duration_labels = ['duration']
	amplitude_labels = ['amplitude']
	slope_dep_labels = ['slope_dep']
	slope_rep_labels = ['slope_rep']

	colors = ['b']

	legends = []


titles = {'spikes':{'labels':n_spikes_labels,'title':n_spikes_title,'unit':n_spikes_unit}
,'duration':{'labels':duration_labels,'title':duration_title,'unit':duration_unit},
'amplitude':{'labels':amplitude_labels,'title':amplitude_title,'unit':amplitude_unit},
'slope_dep':{'labels':slope_dep_labels,'title':slope_dep_title,'unit':slope_unit},
'slope_rep':{'labels':slope_rep_labels,'title':slope_rep_title,'unit':slope_unit},
'slope_dep_max':{'labels':slope_dep_labels,'title':slope_dep_title+" Peak",'unit':slope_unit},
'slope_rep_max':{'labels':slope_rep_labels,'title':slope_rep_title+" Peak",'unit':slope_unit}}

if plot_mode=="complete" or plot_mode=="simple":
	# columns = ['duration','amplitude','slope_rep','slope_dep','spikes']
	# columns = ['duration','amplitude','slope_rep','slope_dep']
	# columns = ['duration','amplitude','slope_dep','slope_rep']
	columns = ['duration','slope_rep','slope_dep','amplitude']
	# columns = ['duration','amplitude','slope_rep_max','slope_dep_max']
	# columns = ['duration','amplitude','slope_dep_max','slope_rep_max']


	# columns = ['duration','amplitude','slope_rep','slope_dep','slope_rep_max','slope_dep_max']
else:
	# columns = [plot_mode]
	columns = plot_mode.split()


# plt.figure(figsize=(15,4*len(columns))) #best size for 4 rows
# cols =1
fig_setup=(cols,len(columns)//cols+len(columns)%cols)
print(fig_setup)
# fig_setup=(2,2)
# plt.figure(figsize=(20,20*fig_setup[1])) 


dirs = sorted(glob.glob(path+"*%s*"%""))
print(dirs)

if(dirs==[]):
	print("Error: No dirs found. Check the extension provided")
	exit()


labels=[]
ignored=0
df_trials = []

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

	# if dir_name == 'exp19':
	# 	print('skiping 19')
	# 	continue
	best_trial = (0,0)
	#Concat all trials from one same experiment day into one df and plots it.
	for j,f in enumerate(files):
		df = pd.read_pickle(f)
		# print(df.describe())
		# print(j,f)

		try:
			dur_means = df[duration_labels].mean()
			laser_diff_pre,laser_diff_pos=get_diffs(dur_means)[1:] #Get duration mean in this Trial
			if(laser_diff_pre>best_trial[0] and laser_diff_pos>best_trial[1] ):
				indx = f.find("exp")
				if(indx >=0):
					trial_id = int(f[indx+3])
				else:
					trial_id = 1
				# df["Trial"]=trial_id
				df_best = df 
				best_trial = (laser_diff_pre,laser_diff_pos)
				best_trial_id = trial_id
				best_trial_f=f[f.find("exp"):]
				# print(j)
		except Exception as e:
			print("Skiping",f)
			print(e)
			pass

	try:
		df_trials.append(df_best)
	except:
		pass


means_df = []
for trial in df_trials:
	# print(trial.describe())
	# print(trial.mean())
	new_df = pd.DataFrame(trial.mean().to_dict(),index=[trial.index.values[-1]])
	# new_df = pd.DataFrame((trial[duration_labels].mean()/trial['control_pre_duration']).to_dict(),index=[trial.index.values[-1]])
	# new_df = pd.DataFrame((trial[duration_labels].mean()/trial['control_pre_duration'].mean()).to_dict(),index=[trial.index.values[-1]])
	# print(trial[duration_labels].mean()/trial['control_pre_duration'])

	# new_df = pd.DataFrame((abs(trial.std()/trial.mean())).to_dict(),index=[trial.index.values[-1]])
	# new_df = pd.DataFrame((trial.std()/trial.mean()).to_dict(),index=[trial.index.values[-1]])
	# print(new_df.describe)
	means_df.append(new_df)
	# print("\n\n\n\n")


means_df = pd.concat(means_df)

columns_in_order = ['control_pre_duration','laser_duration','control_pos_duration',
			'control_pre_amplitude','laser_amplitude','control_pos_amplitude',
			'control_pre_slope_dep','laser_slope_dep','control_pos_slope_dep',
			'control_pre_slope_rep','laser_slope_rep','control_pos_slope_rep']

legend = ['Control pre','Laser','Control pos']
labels = ['Duration'] + ['Amplitude']  + ['Depolarization \n Slope']+ ['Repolarization \n Slope']
labels_indexes = [2,5,8,11]
print(labels)

plt.figure(figsize=(30,20))
plot_boxplot(means_df,columns_in_order,colors,legend,labels,labels_indexes,"Mean value (n=20)")

save_name = path+"general_boxplot_mean_"+save_extension

if save:
	plt.savefig(save_name+".eps",format="eps")
	plt.savefig(save_name+".pdf",format="pdf")
	plt.savefig(save_name+".png",format="png")
if show:
	plt.show()

plt.figure(figsize=(30,20))
for i,var in enumerate(columns):
	pre_label = 'control_pre_'+var
	var_labels = titles[var]['labels']
	var_title = titles[var]['title']

	means_df = []
	for trial in df_trials:
		# new_df = pd.DataFrame((trial[var_labels].mean()/trial[pre_label].mean()).to_dict(),index=[trial.index.values[-1]])
		new_df = pd.DataFrame((trial[var_labels].mean()).to_dict(),index=[trial.index.values[-1]])
		means_df.append(new_df)


	means_df = pd.concat(means_df)

	columns_in_order = var_labels

	legend = ['Control pre','Laser','Control pos']
	# labels = ['Duration'] + ['Amplitude']  + ['Depolarization \n Slope']+ ['Repolarization \n Slope']
	labels = [var_title]
	# labels_indexes = [2,5,8,11]
	labels_indexes = [2]
	print(labels)

	plt.subplot(2,2,i+1)
	plot_boxplot(means_df,columns_in_order,colors,legend,labels,labels_indexes,"Mean value (n=20)")

columns_in_order = []
labels = []

save_name = path+"general_boxplot_grid_mean_"+save_extension

if save:
	plt.savefig(save_name+".eps",format="eps")
	plt.savefig(save_name+".pdf",format="pdf")
	plt.savefig(save_name+".png",format="png")
if show:
	plt.show()

means_df = []
for var in columns:
	pre_label = 'control_pre_'+var
	var_labels = titles[var]['labels']
	var_title = titles[var]['title']

	for trial in df_trials:
		new_df = trial[var_labels]*100/trial[pre_label].mean()
		means_df.append(new_df)

	columns_in_order += var_labels
	labels += [var_title]

	# print(labels)

legend = ['Control pre','Laser','Control pos']
labels_indexes = [2,5,8,11]
means_df = pd.concat(means_df,axis=0)
print(means_df.describe())
plt.figure(figsize=(30,20))
plot_boxplot(means_df,columns_in_order,colors,legend,labels,labels_indexes,"% variation from initial control (n=20)")


# plot_mode+=save_extension
# plot_mode=plot_mode.replace(' ','_')
# print(plot_mode)

save_name = path+"general_boxplot_norm_mean_"+save_extension

if save:
	plt.savefig(save_name+".eps",format="eps")
	plt.savefig(save_name+".pdf",format="pdf")
	plt.savefig(save_name+".png",format="png")
if show:
	plt.show()
