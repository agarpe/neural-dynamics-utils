import sys

sys.path.append("../")
sys.path.append("~/Workspace/scripts/")

import os

directory = os.path.expanduser('~/Workspace/scripts/')
sys.path.append(directory)


from charact_utils import *
from invariant_functions import *
import pandas as pd
import argparse

plt.rcParams.update({'font.size': 17})
plt.rcParams['figure.dpi'] = 300

# fig_format = 'eps'

ap = argparse.ArgumentParser()
ap.add_argument("-cp", "--configpath", required=True, help="Path of config file")
ap.add_argument("-sa", "--save", required=False, default='y', help="Option to save plot file")
ap.add_argument("-sh", "--show", required=False, default='y', help="Option to show plot file")

args = vars(ap.parse_args())

import configparser
Config = configparser.ConfigParser()
config_path = args['configpath']

print('Trying to read config file from %s' % config_path)
if Config.read(config_path) == []:
    print("Error: No config data found")
    exit()


# Get config parameters
parent_config = os.path.dirname(config_path)
print(parent_config)
path = Config.get('signal values', 'path')
path = parent_config + path

n1 = Config.get('signal values', 'n1')
n2 = Config.get('signal values', 'n2')
n3 = Config.get('signal values', 'n3')

dt = Config.getfloat('signal values', 'dt')

ini_rang = Config.getint('burst selection', 'ini_rang')
interval_lim = Config.getint('burst selection', 'end_rang')


file_name = Config.get('output', 'output_path')
fig_format = Config.get('output', 'fig_format')

labels = Config.get('output', 'labels').split(',')
if len(labels) < 3:
    labels += ['']

print(labels)

try:
    color = Config.get('plot', 'color')
    print("Color: ", color)
except:
    pass

show = True if args['show'] == 'y' else False
save = True if args['save'] == 'y' else False

N1_data = read_bursts_events(path + n1)
N2_data = read_bursts_events(path + n2)

if n3 != '':
    N3_data = read_bursts_events(path + n3)
else:
    N3_data = np.array([])


print(len(N1_data), len(N2_data), len(N3_data))
# N1_data, N2_data, N3_data = fix_length(N1_data, N2_data, N3_data)


N1_data = trunc(N1_data, decs=2)
N2_data = trunc(N2_data, decs=2)
N3_data = trunc(N3_data, decs=2)

stats = {}
index = [0]

#get intervals:
N1_interv = get_single_intervals(N1_data)
N2_interv = get_single_intervals(N2_data)
N3_interv = get_single_intervals(N3_data)


N1N2, N2N1 = get_intervals(N1_data, N2_data)
N1N3, N3N1 = get_intervals(N1_data, N3_data)
N2N3, N3N2 = get_intervals(N2_data, N3_data)

period = N1_interv[PER]

# interval_lim = max(N1_interv[DUR]) + max(N2_interv[DUR]) + 20000
# interval_lim = 100000000
# interval_lim = 30000


print(interval_lim)
cycles_to_keep = np.where(period < interval_lim)
print("Removed %d cycles"%(len(period)-len(cycles_to_keep[0])))

all_intervals = [N1_interv, N2_interv, N3_interv, N1N2, N2N1, N1N3, N3N1, N2N3, N3N2]

all_intervals = reduce_intervals(cycles_to_keep, all_intervals)

N1_interv, N2_interv, N3_interv, N1N2, N2N1, N1N3, N3N1, N2N3, N3N2 = all_intervals

period = N1_interv[PER]


print('analyzing single')
N1_interv = analyse(N1_interv, labels[0], stats, index)
N2_interv = analyse(N2_interv, labels[1], stats, index)
N3_interv = analyse(N3_interv, labels[2], stats, index)


print('analyzing pairs')
N1N2, N2N1 = analyse_pair(N1N2, N2N1, labels[0], labels[1], stats, index)
N1N3, N3N1 = analyse_pair(N1N3, N3N1, labels[0], labels[2], stats, index)
N2N3, N3N2 = analyse_pair(N2N3, N3N2, labels[1], labels[2], stats, index)


# plot_intervals(N1_data[:,0],N1_data[:,1])
# plt.show()

# df = pd.DataFrame(stats)

# print(df)
# print(df.keys())


# ran_x = (0,90)
# ran_y = (0,90)
# box_ran = (-0.95,70)

# vert True generates horizontal boxplot. 
plot_intervals_stats(stats, box_ran=None, ignored_intervals=[], vert=False)

output_path = path + file_name
os.makedirs(os.path.dirname(output_path), exist_ok=True)

if save:
    # plt.savefig("./results_invariant_tests/images/"+file_name+"_boxplot.png")
    plt.savefig(output_path + "_boxplot." + fig_format, format=fig_format)
# 
if show:
    plt.show()


# ####################################
# ########  CORRELATIONS  ############
# ####################################


# print(len(period),len(N1_interv[DUR]))
# period = remove_intervals(period, interval_lim)

# N1_interv[DUR] = N1_interv[DUR][cycles_to_keep]
# N2_interv[DUR] = N2_interv[DUR][cycles_to_keep]

print(len(period),len(N1_interv[DUR]),len(N2_interv[DUR]))
print(len(period),len(N1N2[DELAY]),len(N2N1[DELAY]))
print(len(period),len(N1N2[INTERVAL]),len(N2N1[INTERVAL]))

# print(len(cycles_to_keep[0]), len(np.where(N2N1[DELAY][:] < interval_lim)[0]))


# N1N2[INTERVAL] = N1N2[INTERVAL][cycles_to_keep]
# N2N1[INTERVAL] = N2N1[INTERVAL][cycles_to_keep]

# ran_x = ran_x_dur
# ran_y = ran_y_dur

ran_x = False
ran_y = False

try:
    # dur_intervals = [N1_interv[DUR][:-1], N2_interv[DUR][:-1], N3_interv[DUR][:-1]]
    dur_intervals = [N1_interv[DUR], N2_interv[DUR], N3_interv[DUR]]
    if labels is None:
        labels = ['N1', 'N2', 'N3']
except:
    # dur_intervals = [N1_interv[DUR][:-1], N2_interv[DUR][:-1]]
    dur_intervals = [N1_interv[DUR], N2_interv[DUR]]
    # if labels is None:
    #     labels = ['N1', 'N2']

output = path + file_name + "_durations." + fig_format
# 
plot_correlations(period, dur_intervals, labels, "Period (ms)", " Burst Duration (ms)", color='royalblue', save=output, fig_format=fig_format)

# plt.show()

try:
    # pair_intervals = [N1N2[INTERVAL][:-1], N2N1[INTERVAL], N1N3[INTERVAL][:-1], N3N1[INTERVAL], N2N3[INTERVAL][:-1], N3N2[INTERVAL]]
    pair_intervals = [N1N2[INTERVAL], N2N1[INTERVAL], N1N3[INTERVAL], N3N1[INTERVAL], N2N3[INTERVAL], N3N2[INTERVAL]]
    if labels is None:
        labels = ['N1-N2', 'N2-N1', 'N1-N3', 'N3-N1', 'N2-N3', 'N3-N2']
    else:
        labels = [labels[0]+'-'+labels[1], labels[1]+'-'+labels[0], labels[0]+'-'+labels[2], labels[2]+'-'+labels[0], 
                labels[1]+'-'+labels[2], labels[2]+'-'+labels[1]]
except:
    # pair_intervals = [N1N2[INTERVAL][:-1], N2N1[INTERVAL]]
    pair_intervals = [N1N2[INTERVAL], N2N1[INTERVAL]]
    # if labels is None:
    #     labels = ['N1-N2', 'N2-N1']
    # else:
    labels = [labels[0]+'-'+labels[1],labels[1]+'-'+labels[0]]


output = path + file_name + "_intervals." + fig_format

plot_correlations(period, pair_intervals, labels, "Period (ms)", " interval (ms)", color='seagreen', save=output, fig_format=fig_format)

# plt.show()

try:
    # pair_intervals = [N1N2[DELAY][:-1], N2N1[DELAY], N1N3[DELAY][:-1], N3N1[DELAY], N2N3[DELAY][:-1], N3N2[DELAY]]
    pair_intervals = [N1N2[DELAY], N2N1[DELAY], N1N3[DELAY], N3N1[DELAY], N2N3[DELAY], N3N2[DELAY]]

    # if labels is None:
    #     labels = ['N1-N2', 'N2-N1', 'N1-N3', 'N3-N1', 'N2-N3', 'N3-N2']

except:
    # pair_intervals = [N1N2[DELAY][:-1], N2N1[DELAY]]
    pair_intervals = [N1N2[DELAY], N2N1[DELAY]]
    # if labels is None:
    #     labels = ['N1-N2', 'N2-N1']
    # else:
    #     labels = [labels[0]+'-'+labels[1],labels[1]+'-'+labels[0]]


output = path + file_name + "_delays." + fig_format

plot_correlations(period, pair_intervals, labels, "Period (ms)", " delay (ms)", color='brown', save=output, fig_format=fig_format)
# 


# Pair plot
import seaborn as sns

# Convert the stats dict to a DataFrame with the correct labels
# print(stats)
intervals_dict = {}

for key1, inner_dict in stats.items():
    for key2, value in inner_dict.items():
        new_key = f"{key1[1:]}-{key2}"
        intervals_dict[new_key] = value

intervals_df = pd.DataFrame(intervals_dict)



#change font 

# sns.set(rc={'figure.figsize':(60, 60)})

# sns.set(font_scale=20)


pair_plot = sns.pairplot(intervals_df)

plt.savefig(output_path+'_pairplot.pdf', format='pdf')
# plt.savefig(path+file_name+'_pairplot.png', format='png', dpi=300)

# plt.show()





if show:
    plt.show()