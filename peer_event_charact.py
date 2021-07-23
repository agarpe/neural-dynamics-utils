# import charact_utils as utils 
# import argparse 
# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# import scipy.stats as stats
# import scipy.signal as signal


# def plot_hist(data,rang,xlabel,title):
# 	plt.hist(data,rang,width=0.2)
# 	plt.title(title)
# 	plt.xlabel(xlabel)

# def read_n_plot(path1,path2,onoff,title,rang,rang_str,save,show):
# 	spikes1 = utils.read_spike_events(path1,dataview=onoff)
# 	spikes2 = utils.read_spike_events(path2,dataview=onoff)

# 	if spikes1.shape != spikes2.shape:
# 		print("Error: the number of spikes is different")
# 		plt.plot(spikes1,np.ones(spikes1.shape),'.')
# 		plt.plot(spikes2,np.ones(spikes2.shape),'.')
# 		print(spikes1.shape,spikes2.shape)
# 		plt.show()
# 		exit()

# 	diff = spikes1-spikes2

# 	plot_hist(diff,rang,"Event RPD2-VD1 (ms)",title)





# ap = argparse.ArgumentParser()
# ap.add_argument("-p", "--path", required=True, help="Path to the file to analyze")
# ap.add_argument("-r","--range",required=False,default=None, help="Range of data")
# ap.add_argument("-onoff", "--onoff", required=False,default='y', help="Events as on-off")
# ap.add_argument("-sh", "--show", required=False,default='y', help="Show plot")
# ap.add_argument("-sv", "--save", required=False,default='y', help="Save events")
# args = vars(ap.parse_args())


# path = args['path']

# indx=path.rfind("/")
# title = path[indx+1:]

# path1 = path+"_RPD2_events.txt"
# path2 = path+"_VD1_events.txt"

# rang = args['range']
# rang_str = ''
# if rang != None:
# 	rang = [float(r) for r in rang.split()]
# 	rang_str = "range"
# print(rang)

# show= True if args['show']=='y' else False 
# save= True if args['save']=='y' else False 
# onoff=True if args['onoff']=='y' else False 


# read_n_plot(path1,path2,onoff,title,rang,rang_str,save,show)


# if save:
# 	save_path = path+"_peer_event_hist"+rang_str
# 	print(save_path)
# 	plt.savefig(save_path+".eps",format='eps')
# 	plt.savefig(save_path+".png",format='png')
# if show:
# 	plt.show()