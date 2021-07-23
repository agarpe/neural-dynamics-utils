import charact_utils as utils 
import argparse 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import scipy.signal as signal


def plot_hist(data,rang,xlabel,title):
	if str(rang) == 'None':
		plt.hist(data,width=0.2)
	else:
		plt.hist(data,range=rang,width=0.2)
	plt.title(title)
	plt.xlabel(xlabel)

def read_n_plot(path1,path2,onoff,title,rang,rang_str,save,show):
	spikes1 = utils.read_spike_events(path1,dataview=onoff)
	spikes2 = utils.read_spike_events(path2,dataview=onoff)

	if spikes1.shape != spikes2.shape:
		print("Warning: the number of spikes is different")
		print(spikes1.shape,spikes2.shape)
		if show:
			plt.plot(spikes1,np.ones(spikes1.shape),'.')
			plt.plot(spikes2,np.ones(spikes2.shape),'.')
			plt.show()
		# exit()
		n_spikes = min(spikes1.shape[0],spikes2.shape[0])
		print(max(spikes1.shape[0],spikes2.shape[0])-n_spikes,"spikes removed from the end")
		spikes1=spikes1[:n_spikes]
		spikes2=spikes2[:n_spikes]
		print(n_spikes,spikes1.shape,spikes2.shape)
	# else:
	diff = spikes1-spikes2

	plot_hist(diff,rang,"Event RPD2-VD1 (ms)",title)





ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=True, help="Path to the file to analyze")
ap.add_argument("-n", "--names", required=False,default="_control_pre _laser _control_pos", help="Extension names for the files to analyze. Default is control_pre,laser,control_pos")
ap.add_argument("-r","--range",required=False,default=None, help="Range of data")
ap.add_argument("-onoff", "--onoff", required=False,default='n', help="Events as on-off")
ap.add_argument("-nn", "--neu_names",required=True, help="Neuron names separated by white space")

ap.add_argument("-sh", "--show", required=False,default='y', help="Show plot")
ap.add_argument("-sv", "--save", required=False,default='y', help="Save events")
args = vars(ap.parse_args())


path = args['path']
names = args['names'].split()
indx=path.rfind("/")
title = path[indx+1:]

n1,n2 = args['neu_names'].split()

rang = args['range']
rang_str = ''
if str(rang) != 'None':
	rang = [float(r) for r in rang.split()]
	rang_str = "_range" + str(rang)

show= True if args['show']=='y' else False 
save= True if args['save']=='y' else False 
onoff=True if args['onoff']=='y' else False 


plt.figure(figsize=(10*len(names),10))

for i,n in enumerate(names):
	plt.subplot(1,len(names),i+1)
	if onoff:
		path1 = path+n+"_"+n1+"_events.txt"
		path2 = path+n+"_"+n2+"_events.txt"
	else:
		path1 = path+n+"_"+n1+"_single_events.txt"
		path2 = path+n+"_"+n2+"_single_events.txt"

	print(path1,path2)

	read_n_plot(path1,path2,onoff,title+n,rang,rang_str,save=False,show=False)

if save:
	save_path = path+"_peer_event_hist"+rang_str
	print(save_path)
	# plt.savefig(save_path+".eps",format='eps')
	plt.savefig(save_path+".png",format='png')
if show:
	plt.show()