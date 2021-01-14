from superpos_functions import *
from colour import Color
import glob

# rcParams["legend.markerscale"]=10
plt.rcParams.update({'legend.markerscale': 2000})
# plt.rcParams.update({'font.size': 14})

show='n'
stats='y'



import argparse

# 	print("Example: python3 superpos_from_model.py ../../laser_model/HH/data/gna/ gna \"Gna simulation\" 0.001 8 20")
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=True, help="Path to the file to plot")
ap.add_argument("-rp", "--ref_param", required=True, help="Parameter varied during simulations")
ap.add_argument("-ti", "--title", required=False,default="", help="Title of the resulting plot")
ap.add_argument("-dt", "--time_step", required=False, default=0.001, help="Sampling freq of -fs")
ap.add_argument("-wt", "--wind_t", required=False, default=10, help="Half window size in ms")
ap.add_argument("-fl", "--file_limit", required=False, default=-1, help="Limit of files to load")
ap.add_argument("-sa", "--save", required=False, default='y', help="Option to save plot file")
ap.add_argument("-sh", "--show", required=False, default='y', help="Option to show plot file")
ap.add_argument("-st", "--stats", required=False, default='y', help="Option to save stats pkl file")
ap.add_argument("-v", "--verbrose", required=False, default='y', help="Option to verbrose process")
args = vars(ap.parse_args())


path = args['path']
ref_param = args['ref_param'] #name of the parameter varied during simulations
title = args['title']
dt=float(args['time_step'])
t=float(args['wind_t'])
lim=int(args['file_limit']) #number of files limit
show=args['show']
stats=args['stats']
save=args['save']
verb=args['verbrose']


def get_events(f_data,f_events,ms,dt=0.001):
	#read data
	data = pd.read_csv(f_data, delimiter = " ",skiprows=2,header=None)
	data = data.values

	#read events
	events = pd.read_csv(f_events, delimiter = " ",skiprows=2,header=None)
	events = events.values
	points = int(ms /dt)

	waveforms = np.empty((events.shape[0],(points*2)),float)
	if verb =='y':
		print("Waveform shape:",waveforms.shape)
		print("Events shape:",events.shape)

	# print(points)

	time = data[:,0]

	count =0
	for i,event in enumerate(events[:,0]):
		indx = np.where(time == event)[0][0] #finds spike time reference

		try:
			waveforms[i] =data[indx-points:indx+points,1]
		except:
			count +=1
			# print(i)

	# print(count, "events ignored")
	# print(waveforms)
	return waveforms[2:] #Ignore 2 first events, usally artefacts


# files = sorted(os.listdir(path))
files = glob.glob(path+"*")
files.sort(key=os.path.getmtime)
files = files[:lim]

# dt = 0.001
# t = 10 #hh 10ms #vav 10ms

axs = []
labels = []

blue = Color("blue")
# colors = list(red.range_to(Color("green"),len(files)//2))
luminances = np.arange(0.8,0.2,-0.6/len(files))
colors=[]
logs = []
logs_cols = []
log ={}

plt.figure(figsize=(15,20))
for i,f in enumerate(files):
	# print(f)
	# f = path+f
	if(f.find("spikes")==-1 and f.find(".asc")!=-1):

		if verb=='y':
			print(f)
		ref = f.find("Euler")
		f_events = f[:ref]+"spikes_"+f[ref:]
		# print(f_events)

		fs=open(f_events)
		first_line = fs.readline()
		# index =fs.find(".asc")
		# ini = fs.rfind("_")
		fs.close()


		label = ref_param+"="+first_line
		# print(label)
		labels.append(label)

		try:
			trial = get_events(f,f_events,ms=t)
		except:
			print("Error reading events from ",f)
			continue

		if verb=='y':
			print(trial.shape)
		if(trial.shape[0] <=3):
			print("skiping and removing corrupt file")
			print(f)
			os.system("rm "+f+" "+f_events)
			continue

		# color=colors[i%(len(files)//2)].hex_l
		color = blue
		color.luminance = luminances[i%(len(files))]
		color = color.hex_l

		trial =trial[:-1]
		
		ax,ax1,ax2=plot_events(trial,color,tit=title,width_ms=t,dt=0.001,df_log=log,show_durations=False)

		logs_cols.append(label)
		logs.append(log)
		if stats=='y':
			log_df = pd.DataFrame(log)
			log_df.to_pickle(path+ref_param+"-"+first_line[:-3]+"_info.pkl")

		axs.append(ax)
		colors.append(color)

lgnd = plt.legend(axs,labels)
for i,line in enumerate(lgnd.get_lines()):
    line.set_linewidth(2)
    line.set_color(colors[i])

plt.title(title)
plt.xlabel("Time (ms)")
plt.ylabel("Voltage (mV)")

print( path+title+".png")
plt.savefig(path+title+".png")

show = 'n'

if show=='y':
	plt.show()

df = create_dataframe(logs,logs_cols)
# print(df.describe())
df.to_pickle(path+ref_param+"_info.pkl")

