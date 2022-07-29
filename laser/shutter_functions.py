import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import superpos_functions as sf


def savefig(path, path_images, name):
    savepath = path_images + path[path[:-1].rfind('/'):-1] + name
    print("Saving fig at",savepath)
    plt.savefig(savepath + '.png', dpi=200)
    plt.savefig(savepath + '.pdf',format='pdf')


def get_slopes(waveforms,dt=0.1, width_r=40, width_l=40):


    slope_dep =[]
    slope_rep =[]
    for i,spike in enumerate(waveforms):
        spike = sf.align_spike(spike, width_ms_r=width_r,width_ms_l=width_l, dt=dt,id_=i )

        if len(spike) ==0:
            slope_dep.append(0)
            slope_rep.append(0)
            continue

        # vals,th = sf.get_spike_duration(spike,dt=0.1,tol=1)
        slope_inc,slope_dec = sf.get_slope(spike,dt)

        # plt.plot(spike)
        # vals = np.array(vals)
        # plt.plot(vals/0.1,(th,th),'x')
        # plt.title(str(vals[1]-vals[0]))
        # plt.show()
        # try:
        slope_dep.append(slope_inc)
        slope_rep.append(slope_dec)    

        # except:
            # print(vals)
            # continue
    # plt.show()
    return np.array(slope_dep), np.array(slope_rep)

def get_durs(waveforms, width_r=40, width_l=40):

    durations =[]
    for i,spike in enumerate(waveforms):
        spike = sf.align_spike(spike, width_ms_r=width_r,width_ms_l=width_l, dt=0.1,id_=i )

        if len(spike) ==0:
            durations.append(0)
            continue

        vals,th = sf.get_spike_duration(spike,dt=0.1,tol=1)
        # plt.plot(spike)
        # vals = np.array(vals)
        # plt.plot(vals/0.1,(th,th),'x')
        # plt.title(str(vals[1]-vals[0]))
        # plt.show()
        try:
            durations.append(vals[1]-vals[0])
        except:
            print(vals)
            continue

    return np.array(durations)

def read_data(path, ctype):
    waveforms =  sf.read_from_events(path+'_%s_waveform_single.txt'%ctype,max_cols=80/0.1,dt=0.1)

    waveforms = waveforms.values
    print(waveforms.shape)

    try:
        stim_path = path + '_%s_shutter_time_references.txt'%ctype # name from v1
        # stim_path = path[:path.rfind('/')] + 'laser_shutter_time_references.txt'
        print(stim_path)
        stim = np.loadtxt(stim_path)
        print("Stim ",stim.shape)
    except Exception as e:
        stim = []
        if ctype == 'laser':
            print("EXCEPTION:",e.args)
        pass

    return waveforms, stim


def get_metrics_from_file(path, ctype):

    waveforms, stim = read_data(path, ctype)
    durations = get_durs(waveforms)
    slopes_dep, slopes_rep = get_slopes(waveforms)

    try:
        stim = stim[np.where(durations>2)]
    except:
        pass

    durations = durations[np.where(durations>2)]
    slopes_dep = slopes_dep[np.where(durations>2)]
    slopes_rep = slopes_rep[np.where(durations>2)]

    return durations, slopes_dep, slopes_rep, stim


def plot_boxplot(df,column, metric, cut_range, step_range, df_controls=None, df_laser=None):
    
    if cut_range is not None:
        cut_range = [int(r.replace('\\','')) for r in cut_range.split(',')]
    else:
        cut_range = [df[column].min(), df[column].max()]
    # print(cut_range)

    # https://stackoverflow.com/questions/21441259/pandas-groupby-range-of-values
    # print(df[column])
    df["range"] = pd.cut(df[column], np.arange(cut_range[0], cut_range[1], step_range))

    # print(df.describe())

    ax = df.boxplot(column=metric,by='range',figsize=(35,20), showmeans=True, showfliers=False)
    ax.set_ylabel("Spike %s (ms)"%metric)
    ax.set_xlabel("Time %s event (ms)"%column)
    plt.tight_layout()

    if df_controls is not None and df_laser is not None:
        ticks = ax.get_xticklabels()

        ax.boxplot(df_controls["control_%s"%metric][df_controls["control_%s"%metric].notnull()], positions = [-1], showmeans=True, showfliers=False)
        ax.boxplot(df_controls["recovery_%s"%metric][df_controls["recovery_%s"%metric].notnull()], positions = [0], showmeans=True, showfliers=False)
        ax.boxplot(df_laser["laser_%s"%metric][df_laser["laser_%s"%metric].notnull()], positions = [len(ticks)+1], showmeans=True, showfliers=False)

        ticks = ["%s"%t.get_text() for t in ticks] +["control"]+["recovery"] + ["continuous laser"]

        n_ticks = ax.get_xticks()

        ax.set_xticks(n_ticks,ticks)
        ax.set_xticklabels(ticks)

        plt.tight_layout()
    
    # plt.xticks(rotation=45, ha='right')

    return ax

#df dataframe
# column e.g. to_off
# metric e.g. duration
def plot_scatter(df, column, metric, ylabel, xlabel, df_controls=None, median = True):
    plt.figure(figsize=(20,15))
    for name, group in df.groupby("file"):
        fig = plt.plot(group[column], group[metric], marker="o", linestyle="", label=name)
        # print(df_controls.loc[df_controls['file'] == name]["control_duration"])
        if median:
            plt.plot(group[column].median(), group[metric].median(), marker="o", markersize=15, color=fig[-1].get_color(), label= name+' median')
        
        if df_controls is not None:
            control = df_controls.loc[df_controls['file'] == name]["control_duration"]
            recovery = df_controls.loc[df_controls['file'] == name]["recovery_duration"]
            plt.plot(np.zeros(control.size), control, marker="o", linestyle="", color='k')
            plt.plot(np.zeros(recovery.size)+5, recovery, marker="o", linestyle="", color='k')
        
            if median:
                plt.plot(0, control.median(), marker="o", markersize=15, color=fig[-1].get_color())
                plt.plot(0.5, recovery.median(), marker="o", markersize=15, color=fig[-1].get_color())

    plt.legend(fontsize=10)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.tight_layout()


def plot_all_scatter(df, metric, save, df_controls, path, path_images):

    # ## Plot scatter
    # plot_scatter(df, "to_off", metric, "Spike %s (ms)"%metric, "Time from spike to off (ms)")

    # if save:
    #     savefig(path, path_images, "_%s_general_scatter_to_off"%metric)

    # ## Plot scatter on
    # plot_scatter(df, "to_on", metric, "Spike %s (ms)"%metric, "Time from spike to on (ms)")

    # if save:
    #     savefig(path, path_images, "_%s_general_scatter_to_on"%metric)

    ## Plot scatter
    plot_scatter(df, "to_off", metric, "Spike %s (ms)"%metric, "Time from spike to off (ms)", df_controls=df_controls)

    if save:
        savefig(path, path_images, "_%s_general_scatter_to_off_control"%metric)

    ## Plot scatter on
    plot_scatter(df, "to_on", metric, "Spike %s (ms)"%metric, "Time from spike to on (ms)", df_controls=df_controls)

    if save:
        savefig(path, path_images, "_%s_general_scatter_to_on_control"%metric)
