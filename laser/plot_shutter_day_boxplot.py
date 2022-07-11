import os
import argparse
import matplotlib.pyplot as plt

import glob

import superpos_functions as sf
from shutter_functions import *

plt.rcParams.update({'font.size': 35})

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
	day_dict = {'to_on': [],'to_off': [],'duration': [],'repol_slope': [],'depol_slope': [], 'file': []}

	controls_dict = {'control_duration': [], 'recovery_duration': [],'control_depol_slope': [], 'recovery_depol_slope': [], 'control_repol_slope': [], 'recovery_repol_slope': [], 'file': []}

	for file in files:
		file = file[file.rfind('/'):-4]
		file = path + '/events/' + file
		print(file)

		if extension == '' and ('depol' not in file and 'repol' not in file and 'slope' not in file):
			continue
		# exit()
		waveforms, stim = read_data(file, 'laser')
		durations = get_durs(waveforms)
		slopes_dep, slopes_rep = get_slopes(waveforms)
		stim[:,:] *= -1

		print(slopes_dep.shape, slopes_rep.shape, durations.shape)

		stim = stim[np.where(durations>2)]
		slopes_dep = slopes_dep[np.where(durations>2)]
		slopes_rep = slopes_rep[np.where(durations>2)]
		durations = durations[np.where(durations>2)]

		day_dict['to_on'].extend(stim[:,0].tolist())
		day_dict['to_off'].extend(stim[:,1].tolist())
		day_dict['duration'].extend(durations.tolist())
		day_dict['depol_slope'].extend(slopes_dep.tolist())
		day_dict['repol_slope'].extend(slopes_rep.tolist())
				

		day_dict['file'].extend([file]*durations.size)

		control_durations, control_slopes_dep, control_slopes_rep, n = get_metrics_from_file(file, 'control')
		recovery_durations, recovery_slopes_dep, recovery_slopes_rep, n = get_metrics_from_file(file, 'recovery')

		controls_dict['control_duration'].extend(control_durations.tolist())
		controls_dict['recovery_duration'].extend(recovery_durations.tolist())

		controls_dict['control_depol_slope'].extend(control_slopes_dep.tolist())
		controls_dict['recovery_depol_slope'].extend(recovery_slopes_dep.tolist())

		controls_dict['control_repol_slope'].extend(control_slopes_rep.tolist())
		controls_dict['recovery_repol_slope'].extend(recovery_slopes_rep.tolist())

		controls_dict['file'].extend(max([file]*control_durations.size,[file]*recovery_durations.size))

	df_controls = pd.DataFrame.from_dict(controls_dict, orient='index')
	df_controls = df_controls.transpose()

	# print(df_controls.describe())

	df = pd.DataFrame.from_dict(day_dict, orient='index')
	df = df.transpose()

	# print(df.describe())

	df = df.dropna()
	# print(df.describe())

	df_controls.to_pickle(path + path[path[:-1].rfind('/'):-1] +"_shutter_controls.pkl")
	df.to_pickle(path + path[path[:-1].rfind('/'):-1] +"_shutter_laser.pkl")

else:
	df = pd.read_pickle(path + path[path[:-1].rfind('/'):-1] +"_shutter_laser.pkl")
	df_controls = pd.read_pickle(path + path[path[:-1].rfind('/'):-1] +"_shutter_controls.pkl")
# df_laser = pd.read_pickle(path + path[path[:-1].rfind('/'):-1] +"_shutter_laser_continuous.pkl")

#get laser continuous reference
laser_dict = {'laser_duration': [],'laser_depol_slope': [],'laser_repol_slope': [],'file': []}

files_laser = glob.glob(path+"exp*laser*.asc")

for file in files_laser:
	file = file[file.rfind('/'):-4]
	file = path + '/events/' + file
	# print(file)

	# exit()
	laser_durations,laser_depol_slope, laser_repol_slope, n = get_metrics_from_file(file, 'laser')

	laser_dict['laser_duration'].extend(laser_durations.tolist())
	laser_dict['laser_depol_slope'].extend(laser_depol_slope.tolist())
	laser_dict['laser_repol_slope'].extend(laser_repol_slope.tolist())

	laser_dict['file'].extend([file]*laser_durations.size)

df_laser = pd.DataFrame.from_dict(laser_dict, orient='index')
df_laser = df_laser.transpose()

df_laser = df_laser.dropna()
# print(df_laser.describe())

df_laser.to_pickle(path + path[path[:-1].rfind('/'):-1] +"_shutter_laser_continuous.pkl")


if lim != np.inf:
	df.drop(df[df.to_off > lim].index, inplace=True)


df["pulse"] = df["to_on"]-df["to_off"]
df["pulse"] = pd.to_numeric(df["pulse"])
 
path_images = path + '/events/images/shutter/' 
os.system("mkdir -p %s"%path_images)

for metric in ["duration", "depol_slope", "repol_slope"]:

	## Plot boxplot
	# cut_range = args['range']

	# plot_boxplot(df,"to_off", metric, cut_range, step_range)

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

	if save:
		savefig(path, path_images,"_%s_boxplot_control_to_off"%metric)

	## Plot boxplot with control
	cut_range = args['range']

	ax = plot_boxplot(df,"to_on", metric, cut_range, step_range, df_controls, df_laser)

	if save:
		savefig(path, path_images,"_%s_boxplot_control_to_on"%metric)


	## Plot scatter pulse duration
	plot_scatter(df, "pulse", metric, "Spike %s (ms)"%metric, "Pulse duration (ms)", median = False)

	if save:
		savefig(path, path_images,"_general_scatter_pulse_duration")

	## Plot scatter for duration
	plot_all_scatter(df, metric, save, df_controls, path, path_images)
	## Plot duration distribution

	# ## Plot duration distribution
	#################################################
	fig, axes = plt.subplots(1,2)

	axes[0].hist(df["to_on"]-df["to_off"])
	axes[1].bar(list(range(len(df.groupby("file")))),df.groupby("file")["pulse"].mean())

	plt.title("Pulse duration distribution")

	if save:
		savefig(path, path_images, "_pulse_duration")


	# #################################################
	# ## Plot scatter on
	# #################################################
	plt.figure(figsize=(20,15))
	for name, group in df.groupby("file"):
	    p = plt.plot(group["to_on"], group[metric], marker="o", linestyle="", label=name)
	    plt.plot(group["to_off"], group[metric], marker="x", linestyle="", label=name, color=plt.gca().lines[-1].get_color())

	plt.legend()
	plt.ylabel("%s (ms)"%metric)
	plt.xlabel("Time to event (ms)")

	plt.tight_layout()

	if save:
		savefig(path, path_images, "_%s_general_scatter_on_off"%metric)
	# #################################################


if show:
	plt.show()


