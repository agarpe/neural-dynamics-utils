from matplotlib.lines import Line2D
from stats_plot_functions import *


plt.rcParams.update({'font.size': 30})



import argparse

# 	print("Example: python3 superpos_from_model.py ../../laser_model/HH/data/gna/ gna \"Gna simulation\" 0.001 8 20")
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=True, help="Path to the file to show stats from")
# ap.add_argument("-m", "--mode", required=True, 
# 				help="Barchart plot mode: 'simple' for models and 'complete' for experimental. Write columns strings separated by space")
# ap.add_argument("-dt", "--data_type", required=True, help="Whether data is obtain from models or experiments (control-laser-control)")
ap.add_argument("-ch", "--charact", required=False,default='duration', help="Column name in the dataframe with the characteristic to plot")
ap.add_argument("-sb", "--sort_by", required=False,default='time', help="Sort data by 'time' or 'name'")
# ap.add_argument("-s", "--selection", required=False,default='n', help="Spike selection of no_bursts and burst spikes")
ap.add_argument("-pe", "--path_extension", required=False,default="", help="Subpath where pkl files are located. p.e. events")
ap.add_argument("-sa", "--save", required=False, default='n', help="Option to save plot file")
ap.add_argument("-sh", "--show", required=False, default='y', help="Option to show plot file")
ap.add_argument("-v", "--verbrose", required=False, default='n', help="Option to verbrose actions")
ap.add_argument("-log", "--log", required=False, default='n', help="Option to log best trial selection")
args = vars(ap.parse_args())


path = args['path']
charact = args['charact']
# plot_mode = args['mode'] 
# data_type = args['data_type']

ext_path = args['path_extension'] #subpath where 
sort_by = args['sort_by']
show= True if args['show']=='y' else False 
save= True if args['save']=='y' else False 
log= True if args['log']=='y' else False 
verb= True if args['verbrose']=='y' else False 


color_pre = (Color("lightcyan"),Color("cornflowerblue"))
color_pos = (Color("skyblue"),Color("darkblue"))
color_laser = (Color("lightsalmon"),Color("darkred"))




plt.figure(figsize=(20,10))


dirs = sorted(glob.glob(path+"*%s*"%""))
print(dirs)

if(dirs==[]):
	print("Error: No dirs found. Check the extension provided")
	exit()


labels=[]
ignored=0


files = glob.glob(path+"/"+ext_path+"/*.pkl")
if sort_by == 'time':
	files.sort(key=os.path.getmtime)
elif sort_by == 'name':
	files.sort(key=os.path.basename)


best_trial = (0,0)
#Concat all trials from one same experiment day into one df and plots it.
for j,f in enumerate(files):
	print(f)
	df = pd.read_pickle(f)
	# print(df)

	# try:
	# 	# plt.subplots(1,3)
	plt.subplot(3,1,1)
	plot_caract(df,'control_pre_'+charact,"Duration (ms)",color_pre)
	plt.subplot(3,1,2)
	plot_caract(df,'laser_'+charact,"Duration (ms)",color_laser)
	plt.subplot(3,1,3)
	plot_caract(df,'control_pos_'+charact,"Duration (ms)",color_pos)
	# except:
	# 	pass


plt.tight_layout()

# if spike_selection:
# 	plot_mode+="_selection"

if save:
	s_name = path+"_"+charact
	plt.savefig(s_name+".eps",format="eps")
	plt.savefig(s_name+".pdf",format="pdf")
	plt.savefig(s_name+".png",format="png")
if show:
	plt.show()
