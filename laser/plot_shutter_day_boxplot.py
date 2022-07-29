import os
import argparse
import matplotlib.pyplot as plt

import glob

import superpos_functions as sf
from shutter_functions import *

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=True, help="Path to the file to show stats from")
ap.add_argument("-ex", "--extension", required=False, default='', help="Extension in the file, such as 'depol' in 'exp1_depol_30.asc'")
ap.add_argument("-bin", "--bin-size", required=False, default=50, help="Bin size range")
ap.add_argument("-sa", "--save", required=False, default='y', help="Option to save plot file")
ap.add_argument("-sh", "--show", required=False, default='y', help="Option to show plot file")
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


read = len(glob.glob(path+path[path[:-1].rfind('/'):-1]+"_shutter*.pkl")) > 0

if not read:
	files = glob.glob(path+"exp*%s*.asc"%extension)
	files.sort(key=os.path.getmtime)

	# day_dict = {'to_off': np.empty(1),'duration': np.empty(1)}
	day_dict = {'to_on': [],'to_off': [],'duration': [], 'file': []}

	controls_dict = {'control_duration': [], 'recovery_duration': [], 'file': []}

	for file in files:
		file = file[file.rfind('/'):-4]
		file = path + '/events/' + file
		print(file)

		# exit()
		waveforms, stim = read_data(file, 'laser')
		durations = get_durs(waveforms)

		stim = stim[np.where(durations>2)]
		durations = durations[np.where(durations>2)]

		day_dict['to_on'].extend(stim[:,1].tolist())
		day_dict['to_off'].extend(stim[:,1].tolist())
		day_dict['duration'].extend(durations.tolist())
		day_dict['file'].extend([file]*durations.size)

		control_durations, n = get_durs_from_file(file, 'control')
		recovery_durations, n = get_durs_from_file(file, 'recovery')

		controls_dict['control_duration'].extend(control_durations.tolist())
		controls_dict['recovery_duration'].extend(recovery_durations.tolist())
		controls_dict['file'].extend(max([file]*control_durations.size,[file]*recovery_durations.size))

	df_controls = pd.DataFrame.from_dict(controls_dict, orient='index')
	df_controls = df_controls.transpose()

	print(df_controls.describe())

	df = pd.DataFrame.from_dict(day_dict, orient='index')
	df = df.transpose()

	print(df.describe())

	df = df.dropna()
	print(df.describe())


	df_controls.to_pickle(path + path[path[:-1].rfind('/'):-1] +"_shutter_controls.pkl")
	df.to_pickle(path + path[path[:-1].rfind('/'):-1] +"_shutter_laser.pkl")

else:
	df = pd.read_pickle(path + path[path[:-1].rfind('/'):-1] +"_shutter_laser.pkl")
	df_controls = pd.read_pickle(path + path[path[:-1].rfind('/'):-1] +"_shutter_controls.pkl")

#################################################
## Plot boxplot
#################################################

# https://stackoverflow.com/questions/21441259/pandas-groupby-range-of-values
print(df["to_off"])
df["range"] = pd.cut(df["to_off"], np.arange(df["to_off"].min(), df["to_off"].max(), 50))

print(df.describe())

ax = df.boxplot(column='duration',by='range',figsize=(20,15), showmeans=True)
ax.set_ylabel("Spike Duration (ms)")
ax.set_xlabel("Time to off event (ms)")
plt.tight_layout()


if save:
	path_images = path + '/events/images/shutter/' 
	os.system("mkdir -p %s"%path_images)
	savepath = path_images + path[path[:-1].rfind('/'):-1] + "_boxplot"
	print("Saving fig at",savepath)
	plt.savefig(savepath + '.png')
	plt.savefig(savepath + '.pdf',format='pdf')
#################################################


#################################################
## Plot boxplot with control
#################################################

# https://stackoverflow.com/questions/21441259/pandas-groupby-range-of-values
print(df["to_off"])
df["range"] = pd.cut(df["to_off"], np.arange(df["to_off"].min(), df["to_off"].max(), 50))

print(df.describe())

ax = df.boxplot(column='duration',by='range',figsize=(25,15), showmeans=True)
ax.set_ylabel("Spike Duration (ms)")
ax.set_xlabel("Time to off event (ms)")
ticks = ax.get_xticklabels()

ax.boxplot(df_controls["control_duration"][df_controls["control_duration"].notnull()], positions = [-1], showmeans=True)
ax.boxplot(df_controls["recovery_duration"][df_controls["recovery_duration"].notnull()], positions = [0], showmeans=True)

ticks = ["%s"%t.get_text() for t in ticks] +["control"]+["recovery"]

n_ticks = ax.get_xticks()

ax.set_xticks(n_ticks,ticks)
ax.set_xticklabels(ticks)

plt.tight_layout()

if save:
	path_images = path + '/events/images/shutter/' 
	os.system("mkdir -p %s"%path_images)
	savepath = path_images + path[path[:-1].rfind('/'):-1] + "_boxplot_control"
	print("Saving fig at",savepath)
	plt.savefig(savepath + '.png')
	plt.savefig(savepath + '.pdf',format='pdf')



#################################################

#################################################
## Plot scatter
#################################################
plt.figure(figsize=(20,15))
for name, group in df.groupby("file"):
    plt.plot(group["to_off"], group["duration"], marker="o", linestyle="", label=name)
plt.legend()
plt.ylabel("Duration (ms)")
plt.xlabel("Time to off event (ms)")

if save:
	savepath = path_images + path[path[:-1].rfind('/'):-1] + "_general_scatter"
	print("Saving fig at",savepath)
	plt.savefig(savepath + '.png')
	plt.savefig(savepath + '.pdf',format='pdf')
#################################################


if show:
	plt.show()


