from superpos_functions import *
from colour import Color
import glob

# rcParams["legend.markerscale"]=10
plt.rcParams.update({'legend.markerscale': 2000})
# plt.rcParams.update({'font.size': 14})


if len(sys.argv) >6:
	path = sys.argv[1]
	ref_param = sys.argv[2] #name of the parameter varied during simulations
	title = sys.argv[3]
	dt=float(sys.argv[4])
	t=float(sys.argv[5])
	lim=int(sys.argv[6]) #number of files limit

elif len(sys.argv) >3:
	path = sys.argv[1]
	ref_param = sys.argv[2] #name of the parameter varied during simulations
	title = sys.argv[3]
	dt=float(sys.argv[4])
	t=float(sys.argv[5])
	lim=-1
elif len(sys.argv) >2:
	path = sys.argv[1]
	ref_param = sys.argv[2]
	title = ""
	t=10 #hh 10ms #vav 10ms
	dt=0.001
	lim=-1
	# path_spk = sys.argv[2]
else:
	print("Use: python3 superpos_from_model.py path ref_param title [dt] [win_t] [max_files]")
	print("Example: python3 superpos_from_model.py ../../laser_model/HH/data/gna/ gna \"Gna simulation\" 0.001 8 20")
	exit()


def get_events(f_data,f_events,ms,dt=0.001):
	#read data
	data = pd.read_csv(f_data, delimiter = " ",skiprows=2,header=None)
	data = data.values

	#read events
	events = pd.read_csv(f_events, delimiter = " ",skiprows=2,header=None)
	events = events.values
	points = int(ms /dt)

	waveforms = np.empty((events.shape[0],(points*2)),float)
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
logs = {}
ampl_log = []

plt.figure(figsize=(15,20))
for i,f in enumerate(files):
	# print(f)
	# f = path+f
	if(f.find("spikes")==-1 and f.find(".asc")!=-1):

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


		trial = get_events(f,f_events,ms=t)
		print(trial.shape)
		if(trial.shape[0] <=3):
			print("skiping and removing corrupt file")
			os.system("rm "+f+" "+f_events)
			continue

		# color=colors[i%(len(files)//2)].hex_l
		color = blue
		color.luminance = luminances[i%(len(files))]
		color = color.hex_l

		trial =trial[:-1]
		
		ax,ax1,ax2=plot_events(trial,color,tit=title,width_ms=t,dt=0.001,duration_log=ampl_log,show_durations=False)
		# ax,ax1,ax2=plot_events(trial,color,tit=title,width_ms=t,dt=0.001)
		# try:
		# 	logs['spike'].append(ampl_log[:],axis=0)
		# 	# print("success")
		# except:
			# logs['spike']=ampl_log[:]

		logs['']=ampl_log[:]

		# logs = logs.append(ampl_log[:])
		# print(logs)
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
# plt.show()
# plt.clf()

# print(logs)

# df = pd.DataFrame(logs)
df =pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in logs.items() ]))
# df = pd.DataFrame(logs)
# print(df)
print(df.describe())
print(df.mean())
#Saving duration dataframes

df.to_pickle(path+title+"_info.pkl")


#boxplot
# df.boxplot(grid=False,figsize=(10,30),fontsize=20)
# plt.xlabel(ref_param)
# plt.ylabel("Spike width (ms)")
# plt.savefig(path +title+"boxplots.png")
# plt.show()
