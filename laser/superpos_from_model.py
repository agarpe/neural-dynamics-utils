from superpos_functions import *
from colour import Color
import glob

# rcParams["legend.markerscale"]=10
plt.rcParams.update({'legend.markerscale': 2000})
# plt.rcParams.update({'font.size': 14})



if len(sys.argv) >2:
	path = sys.argv[1]
	title = sys.argv[2]
elif len(sys.argv) >1:
	path = sys.argv[1]
	title = ""
	# path_spk = sys.argv[2]
else:
	print("Use: python3 superpos_from_model.py path [title]")
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
	print(waveforms.shape)
	print(events.shape)

	# print(points)

	time = data[:,0]

	count =0
	for i,event in enumerate(events[:,0]):
		indx = np.where(time == event)[0][0] #finds spike time reference

		try:
			waveforms[i] =data[indx-points:indx+points,1]
		except:
			count +=1
			print(i)

	print(count, "events ignored")
	# print(waveforms)
	return waveforms


# files = sorted(os.listdir(path))
files = glob.glob(path+"*")
files.sort(key=os.path.getmtime)
# files = files[:6]

dt = 0.001
t = 10 #hh 10ms #vav 10ms

axs = []
labels = []

blue = Color("blue")
# colors = list(red.range_to(Color("green"),len(files)//2))
luminances = np.arange(0.8,0.2,-0.6/len(files))
colors=[]
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

		label = "tau="+first_line
		# print(label)
		labels.append(label)


		trial = get_events(f,f_events,ms=t)

		# color=colors[i%(len(files)//2)].hex_l
		color = blue
		color.luminance = luminances[i%(len(files))]
		color = color.hex_l

		trial =trial[:-1]
		
		ax,ax1,ax2=plot_events(trial,color,tit=title,width_ms=t,dt=0.001)
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






