import matplotlib.colors as mcolors
import matplotlib.cm as cm
from superpos_functions import *
from colour import Color
import glob

# rcParams["legend.markerscale"]=10
plt.rcParams.update({'legend.markerscale': 40})
plt.rcParams.update({'font.size': 50})

show='n'
stats='y'



import argparse

# 	print("Example: python3 superpos_from_model.py ../../laser_model/HH/data/gna/ gna \"Gna simulation\" 0.001 8 20")
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=True, help="Path to the file to plot")
ap.add_argument("-rp", "--ref_param", required=True, help="Parameter varied during simulations")
ap.add_argument("-ti", "--title", required=False,default="", help="Title of the resulting plot")
ap.add_argument("-dt", "--time_step", required=False, default=0.001, help="Sampling freq of -fs")
ap.add_argument("-c", "--color", required=False, default="blue", help="Color for plot")
ap.add_argument("-wt", "--wind_t", required=False, default=10, help="Half window size in ms")
ap.add_argument("-xl", "--x_lim", required=False, default='', help="Plot xlim values separated by white space")
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
show = True if args['show']=='y' else False 
save = True if args['save']=='y' else False 
stats = True if args['stats']=='y' else False 
verb= True if args['verbrose']=='y' else False 

color = args['color']

xlim = args['x_lim']
if xlim != '':
	xlim = [float(l) for l in xlim.split()]


# lim= 25


def get_events(f_data,f_events,ms,dt=0.001):
	#read data
	data = pd.read_csv(f_data, delimiter = " ",skiprows=2,header=None)
	data = data.values

	#read events
	events = pd.read_csv(f_events, delimiter = " ",skiprows=2,header=None)
	events = events.values
	points = int(ms /dt)

	waveforms = np.empty((events.shape[0],(points*2)),float)
	if verb :
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
	return waveforms[2:-2] #Ignore 2 first events, usally artefacts

if verb:
	print(path)
# files = sorted(os.listdir(path))
files = sorted(glob.glob(path+"*"))
# files.sort(key=os.path.getmtime)
files = files[:lim]




print(files)
# dt = 0.001
# t = 10 #hh 10ms #vav 10ms

if len(files) == 0:
	print("No asc files found")
	exit()


axs = []
labels = []

blue = Color(color)
# colors = list(red.range_to(Color("green"),len(files)//2))
luminances = np.arange(0.9,0.1,-0.7/len(files))
colors=[]
logs = []
logs_cols = []
log ={}

nValues =[]
# normalize = mcolors.Normalize(vmin=nValues.min(), vmax=nValues.max())
# colormap = cm.jet

fig = plt.figure(figsize=(18,20))
for i,f in enumerate(files):
	# print(f)
	# f = path+f
	if(f.find("spikes")==-1 and f.find(".asc")!=-1):

		if verb:
			print(f)

		ref = f.find("Euler")
		if(ref!=-1):
			f_events = f[:ref]+"spikes_"+f[ref:]
		else:
			f_events = f[:-4]+"_spikes.asc"



		# print(f_events)

		fs=open(f_events)
		first_line = fs.readline()
		# index =fs.find(".asc")
		# ini = fs.rfind("_")
		fs.close()


		label = ref_param+"="+first_line
		print(label)
		labels.append(label)
		print(first_line)
		nValues.append(float(first_line))

		try:
			trial = get_events(f,f_events,ms=t)
		except Exception as ex:
			print("Error reading events from ",f)
			print(type(ex).__name__, ex.args)
			print("skiping and removing corrupt file")
			print(f)
			os.system("rm "+f+" "+f_events)
			continue

		if verb:
			print(trial.shape)
		# if(trial.shape[0] <=8):
		# 	print("skiping and removing corrupt file")
		# 	print(f)
		# 	os.system("rm "+f+" "+f_events)
		# 	continue

		# color=colors[i%(len(files)//2)].hex_l
		color = blue
		color.luminance = luminances[i%(len(files))]
		color = color.hex_l
		# color=colormap(normalize(n))


		trial =trial[:-1]
		
		ax,ax1,ax2=plot_events(trial,color,tit=title,width_ms=t,dt=dt,df_log=log,show_durations=False)

		logs_cols.append(label)
		logs.append(log)
		if stats:
			try:
				log_df = pd.DataFrame(log)
				log_df.to_pickle(path+ref_param+"-"+first_line[:-3]+"_info.pkl")

			except ValueError as e:
				print("Passing data ",i,"with length",[len(log[x]) for x in log if isinstance(log[x], list)])
				print("Exception:",e)
				pass
			except Exception as e:
				print("Passing data",i,e)
				pass

		axs.append(ax)
		colors.append(color)

# lgnd = plt.legend(axs,labels)
# for i,line in enumerate(lgnd.get_lines()):
#     line.set_linewidth(2)
#     line.set_color(colors[i])

# scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
# scalarmappaple.set_array(nValues)
# plt.colorbar(scalarmappaple)
# # plt.colorbar();

cmap = mcolors.LinearSegmentedColormap.from_list('my_cmap', colors)
norm = mcolors.Normalize(min(nValues), max(nValues))
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
fig.colorbar(sm)

if xlim != '':
# plt.xlim([20,90])
	plt.xlim(xlim)

plt.title(title)
plt.xlabel("Time (ms)")
plt.ylabel("Voltage (mV)")
plt.tight_layout()

# print( path+title+".png")
# plt.savefig(path+title+".png")
save_name = path+title+"_"+args['color']
print( save_name+".eps")
plt.savefig(save_name+".eps",format="eps")
print( save_name+".png")
plt.savefig(save_name+".png",format="png")

if show:
	plt.show()

df = create_dataframe(logs,logs_cols)
print(df.describe())

os.system("mkdir -p %s/general"%path)
df.to_pickle(path+"/general/"+ref_param+"_info.pkl")

