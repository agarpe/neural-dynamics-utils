import os
import argparse
import matplotlib.pyplot as plt

import glob

import superpos_functions as sf
from shutter_functions import *

plt.rcParams.update({'font.size': 25})

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=True, help="Path to the file to show stats from")
ap.add_argument("-ex", "--extension", required=False, default='', help="Extension in the file, such as 'depol' in 'exp1_depol_30.asc'")
ap.add_argument("-bin", "--bin-size", required=False, default=50, help="Bin size range")
ap.add_argument("-sa", "--save", required=False, default='y', help="Option to save plot file")
ap.add_argument("-sh", "--show", required=False, default='y', help="Option to show plot file")
ap.add_argument("-pkl", "--pkl", required=False, default='y', help="Read from pkl if existing")
ap.add_argument("-rang","--range", required=False, default=None, help="Cut range for boxplot")
ap.add_argument("-rastep","--range_step", required=False, default=50, help="Step for distance chunks")
ap.add_argument("-lim","--limit", required=False, default=np.inf, help="Time limit to beginning of shutter event")
# ap.add_argument("-v", "--verbrose", required=False, default='n', help="Option to verbrose actions")
# ap.add_argument("-log", "--log", required=False, default='n', help="Option to log best trial selection")
args = vars(ap.parse_args())


path = args['path']
extension = args['extension'] #subpath where 

# sort_by = args['sort_by']
show= True if args['show']=='y' else False 
save= True if args['save']=='y' else False 
# log= True if args['log']=='y' else False 
# verb= True if args['verbrose']=='y' else False 

step_range = int(args['range_step'])

lim = float(args['limit'])

read = args['pkl'] != 'y' or (not len(glob.glob(path+path[path[:-1].rfind('/'):-1]+"_shutter*.pkl")) > 0)

if read:
	print("Reading from asc")
	files = glob.glob(path+"exp*%s*.asc"%extension)
	files.sort(key=os.path.getmtime)
	print(files)

	if len(files) == 0:
		print("Files not found for path:", path) 
		exit() 


	# day_dict = {'to_off': np.empty(1),'duration': np.empty(1)}
	# day_dict = {'to_on': [],'to_off': [],'duration': [],'repol_slope': [],'depol_slope': [], 'file': []}
	day_dict = {'to_on': [],'to_off': [],'duration': [],'repol_slope': [],'depol_slope': [],'repol_slope2': [],'depol_slope2': [], 'file': []}

	# controls_dict = {'control_duration': [], 'recovery_duration': [],'control_depol_slope': [], 'recovery_depol_slope': [], 'control_repol_slope': [], 'recovery_repol_slope': [], 'file': []}
	controls_dict = {'control_to_on': [], 'control_to_off': [], 'control_duration': [],'control_depol_slope': [], 'control_repol_slope': [],
	'control_depol_slope2': [], 'control_repol_slope2': [], 'file': [], 'stim':[]}
	
	recovery_dict = {'control_to_on': [], 'control_to_off': [], 'control_duration': [],'control_depol_slope': [], 'control_repol_slope': [],
	'control_depol_slope2': [], 'control_repol_slope2': [], 'file': [], 'stim':[]}
	
	# {'recovery_to_on': [], 'recovery_to_off': [], 'control_to_on': [], 'control_to_off': [], 'control_duration': [], 'recovery_duration': [],'control_depol_slope': [], 'recovery_depol_slope': [], 'control_repol_slope': [], 'recovery_repol_slope': [],
	# 'recovery_depol_slope2': [], 'control_repol_slope2': [], 'recovery_repol_slope2': [], 'file': [], 'stim':[]}

	# files = files[:4]

	for file in files:
		file_ext = file[file.rfind('/'):-4]
		file = path + '/events/' + file_ext
		print(file)

		if extension == '' and ('depol' not in file and 'repol' not in file and 'slope' not in file):
			continue
		# exit()
		plt.figure()
		bunched_data = get_metrics_from_file(file, 'laser', slope_position=0.98, dict_=day_dict)
		plt.title('shutter laser\n'+file_ext)
		plt.tight_layout()
		print(path+"events/images/charact_"+file_ext[1:-4]+'_laser')
		if save:
			plt.savefig(path+"events/images/charact_"+file_ext[1:-4]+'_laser')
		# plt.show()

		plt.figure()
		bunched_data_control = get_metrics_from_file(file, 'control', slope_position=0.99, dict_=controls_dict, ext='control_')
		
		print(bunched_data_control[0])

		plt.title('control\n'+file_ext)
		plt.tight_layout()
		if save:
			plt.savefig(path+"events/images/charact_"+file_ext[1:-4]+'_control')
		
		plt.figure()
		bunched_data_recovery = get_metrics_from_file(file, 'recovery', slope_position=0.99, dict_=recovery_dict, ext='control_')
		plt.title('recovery\n'+file_ext)
		plt.tight_layout()
		if save:
			plt.savefig(path+"events/images/charact_"+file_ext[1:-4]+'_recovery')
		# plt.show()
		print(bunched_data_recovery[0])
	
	df_controls = pd.DataFrame.from_dict(controls_dict, orient='index')
	df_controls = df_controls.transpose()

	df_recovery = pd.DataFrame.from_dict(recovery_dict, orient='index')
	df_recovery = df_recovery.transpose()

	df = pd.DataFrame.from_dict(day_dict, orient='index')
	df = df.transpose()

	# print(df.describe())

	# df = df.dropna()
	print(df.describe())

	df_controls.to_pickle(path + path[path[:-1].rfind('/'):-1] +"_shutter_controls.pkl")
	df_recovery.to_pickle(path + path[path[:-1].rfind('/'):-1] +"_shutter_recovery.pkl")
	df.to_pickle(path + path[path[:-1].rfind('/'):-1] +"_shutter_laser.pkl")

else:
	df = pd.read_pickle(path + path[path[:-1].rfind('/'):-1] +"_shutter_laser.pkl")
	df_controls = pd.read_pickle(path + path[path[:-1].rfind('/'):-1] +"_shutter_controls.pkl")
	df_recovery = pd.read_pickle(path + path[path[:-1].rfind('/'):-1] +"_shutter_recovery.pkl")
# df_laser = pd.read_pickle(path + path[path[:-1].rfind('/'):-1] +"_shutter_laser_continuous.pkl")


# exit()
#get laser continuous reference
laser_dict = {'laser_to_on':[], 'laser_to_off':[], 'stim':[],'laser_duration': [],'laser_depol_slope': [],'laser_repol_slope': [],
				'laser_depol_slope2': [],'laser_repol_slope2': [],'file': []}
# exit()

files_laser = glob.glob(path+"exp*laser*.asc")

for file in files_laser:
	file_ext = file[file.rfind('/'):-4]
	file = path + '/events/' + file_ext
	# print(file)

	# exit()
	plt.figure()
	bunched_data_laser = get_metrics_from_file(file, 'laser', slope_position=0.99, ext='laser_', dict_=laser_dict)

	plt.title('laser\n'+file_ext)
	print("saving ",path+'laser_'+file_ext[1:])
	plt.tight_layout()
	if save:
		plt.savefig(path+"events/images/charact_"+file_ext[1:-4]+'_laser')

df_laser = pd.DataFrame.from_dict(laser_dict, orient='index')
df_laser = df_laser.transpose()

# df_laser = df_laser.dropna() #Warning: breaks df with the "stim and to_on/off" empty
# print(df_laser.describe())

df_laser.to_pickle(path + path[path[:-1].rfind('/'):-1] +"_shutter_laser_continuous.pkl")


if lim != np.inf:
	df.drop(df[df.to_off > lim].index, inplace=True)


df["pulse"] = df["to_on"]-df["to_off"]
df["pulse"] = pd.to_numeric(df["pulse"])

if args['range'] is not None:
	cut_range = args['range'].replace('\\','').split(',')

	df.drop(df[df.to_off < int(cut_range[0])].index, inplace=True)
	df.drop(df[df.to_off > int(cut_range[1])].index, inplace=True)
 
ext = "ext"+extension+"_"

path_own = ext + str(args['range_step']) + str(args['range'].replace('\\','') if args['range'] is not None else '' + "_" + str(step_range))
path_images = path + '/events/images/shutter/' +path_own +'/'+path_own
os.system("mkdir -p %s"%path_images)

metrics = ["duration", "depol_slope", "repol_slope", "depol_slope2", "repol_slope2"]
for metric in metrics:

	cut_range = args['range']

	plot_boxplot_mean(df,"to_off",metric, cut_range, step_range, df_controls)
	
	if save:
		savefig(path, path_images, "_%s_mean_boxplot_to_off"%metric)

	plot_boxplot_mean(df,"to_off",metric, cut_range, step_range, df_recovery)
	
	exit()

	# # Plot boxplot
	# cut_range = args['range']

	# plot_boxplot(df,"to_off", 'diff_'+metric, cut_range, step_range)

	# if save:
	# 	savefig(path, path_images, "_%s_boxplot_to_off"%metric)


	# ## Plot boxplot to on
	# cut_range = args['range']
	# plot_boxplot(df,"to_on", metric, cut_range, step_range)
	# if save:
	# 	path_images = path + '/events/images/shutter/' 
	# 	os.system("mkdir -p %s"%path_images)
	# 	savefig(path, path_images, "_%s_boxplot_to_on"%metric)

	#################################################
	## Plot boxplot with control
	cut_range = args['range']

	ax = plot_boxplot(df,"to_off", metric, cut_range, step_range, df_controls, df_laser)
	plt.suptitle(path_images)
	if save:
		savefig(path, path_images,"_%s_boxplot_control_to_off"%(metric))
	# plt.show()
	# continue
	# exit()
	# ## Plot boxplot with control to on
	# cut_range = args['range']

	# ax = plot_boxplot(df,"to_on", metric, cut_range, step_range, df_controls, df_laser)

	# if save:
	# 	savefig(path, path_images,"_%s_boxplot_control_to_on"%(metric))


	# ## Plot scatter pulse duration
	# plot_scatter(df, "pulse", metric, "Spike %s (ms)"%metric, "Pulse duration (ms)", median = False)

	# if save:
	# 	savefig(path, path_images,"_general_scatter_pulse_duration")

	## Plot scatter for duration
	plot_all_scatter(df, metric, save, df_controls, path, path_images)
	## Plot duration distribution

	# # ## Plot duration distribution
	# #################################################
	# fig, axes = plt.subplots(1,2)

	# axes[0].hist(df["to_on"]-df["to_off"])
	# axes[1].bar(list(range(len(df.groupby("file")))),df.groupby("file")["pulse"].mean())

	# plt.title("Pulse duration distribution")

	# if save:
	# 	savefig(path, path_images, "_pulse_duration")


	# # #################################################
	# # ## Plot scatter on
	# # #################################################
	# plt.figure(figsize=(20,15))
	# for name, group in df.groupby("file"):
	#     p = plt.plot(group["to_on"], group[metric], marker="o", linestyle="", label=name)
	#     plt.plot(group["to_off"], group[metric], marker="x", linestyle="", label=name, color=plt.gca().lines[-1].get_color())

	# plt.legend()
	# plt.ylabel("%s (ms)"%metric)
	# plt.xlabel("Time to event (ms)")

	# plt.tight_layout()

	# if save:
	# 	savefig(path, path_images, "_%s_general_scatter_on_off"%metric)
	# # #################################################


if show:
	plt.show()


