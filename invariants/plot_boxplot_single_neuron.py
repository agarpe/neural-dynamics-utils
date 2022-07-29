import sys

sys.path.append("../")
sys.path.append("~/Workspace/scripts/")

from charact_utils import *
# from invariant_functions import *
import pandas as pd
import argparse

plt.rcParams.update({'font.size': 37})
plt.rcParams['figure.dpi'] = 300

# fig_format = 'eps'

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=True, help="Patern path of events files")
ap.add_argument("-pf", "--prefix", required=True, help="File prefix")
ap.add_argument("-n", "--neuron", required=True, help="Neuron representing phase1")
ap.add_argument("-l", "--labels", required=False, cadefault="N1,N2,N3", help="Neurons labels e.g. N1,N3  or  N1,N2,N3")
ap.add_argument("-ff", "--fig_format", required=False, default='eps', help="Figure output format")
ap.add_argument("-ti", "--title", required=False, default=None, help="Title of the resulting plot")
ap.add_argument("-sa", "--save", required=False, default='y', help="Option to save plot file")
ap.add_argument("-sh", "--show", required=False, default='y', help="Option to show plot file")
args = vars(ap.parse_args())


path = args['path']
n1 = args['neuron']
# interval_lim = int(args['interval_lim'])

fig_format = args['fig_format']

labels = args['labels'].split(',')
if len(labels) < 3:
    labels += ['']
print(labels)

file_name = args['prefix']

title = args['title']

show = True if args['show'] == 'y' else False
save = True if args['save'] == 'y' else False

N1_data = read_bursts_events(path + n1)
print(len(N1_data))

N1_data = trunc(N1_data, decs=2)

stats = {}
index = [0]

#get intervals:
N1_interv = get_single_intervals(N1_data)

period = N1_interv[PER]


print('analyzing single')
N1_interv = analyse(N1_interv, labels[0], stats, index)

plot_intervals_stats(stats, box_ran=(100,1800),ignored_intervals = [], title=title)

#
if save:
    # plt.savefig("./results_invariant_tests/images/"+file_name+"_boxplot.png")
    plt.savefig(path + file_name + "_boxplot." + fig_format, format=fig_format)

if show:
    plt.show()

