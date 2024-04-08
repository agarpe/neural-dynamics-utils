from charact_utils import *



#text_pos inverts 
def plot_corr(ax,x,y,title1,title2,ran_x,ran_y,show=True,color='b',text_pos_y=0,text_pos_x=0,text_pos=0):

    r_sq, Y_pred, slope = do_regression(x, y, title2, False)

    # if(ran_y != False):
    #     max_ = ran_y[1]
    #     ax.ylim(ran_y)
    #     # plt.text(max_-(0.25*max_)*text_pos_x,max_-(0.25*max_)*text_pos_y,"R² = "+"{:6f}".format(r_sq))
        # x_text, y_text = get_pos(text_pos, ran_x, ran_y)
        # ax.text(x_text - text_pos_x, y_text - text_pos_y, "R² = {:.4f}".format(r_sq))

    ax.text(0.15, 0.8, "R² = {:.4f}".format(r_sq), transform=ax.transAxes)


    if(ran_x != False):
        ax.xlim(ran_x)

    ax.set_title(title2[:-4])

    #maroon
    ax.plot(x, Y_pred, color='grey')
    ax.plot(x,y,'.',color=color)
    # plt.text(1,max_-0.25*max_,"R² = "+str(r_sq)[:8])
    # plt.text(1,max_-0.5*max_,"slope = "+str(slope)[:8])

    ax.set_xlabel(title1)
    ax.set_ylabel(title2)
    if show:
        plt.show()



def plot_correlations(x_data, ys_data, neu_labels, x_label, y_label, ran_x = False, ran_y = False, color='b',save=None, fig_format='png'):
    cols = len(ys_data)
    rows = cols//3 + 1 
    # print(rows,cols)

    cols = len(ys_data)//rows
    # print(rows,cols)
    fig, axes = plt.subplots(nrows=rows, ncols=cols, sharex=True, sharey=True, num=1, clear=True)


    for i,(interval,neu_label) in enumerate(zip(ys_data,neu_labels)):
        print(neu_label)

        if axes.ndim > 1:
            ax = axes[i//3][i%3]
        else:
            ax = axes[i]

        plot_corr(ax, x_data, interval, x_label, neu_label + y_label, ran_x, ran_y, False, color=color)

    plt.tight_layout()

    if save is not None:
        plt.savefig(save, format=fig_format)






def remove_intervals(intervals, max_lim):
    ids = np.where(intervals < max_lim)

    return intervals[ids]

def reduce_intervals(cycles, intervals):
    for i, interval in enumerate(intervals):
        for j, inter in enumerate(interval):
            intervals[i][j] = inter[cycles]


    return intervals


def plot_intervals(inis,ends):

    print(inis.shape)
    plt.plot(inis, np.ones(len(inis)), '|')
    plt.plot(ends, np.ones(len(ends)), '|')

# Plot boxplot for given Intervals.
# stats: dict with interval for each neuron. 
def plot_intervals_stats(stats, box_ran, norm=False, pos=False, ignored_intervals=["IBI"], title=None, figsize=(20,17), vert=False):
    if not vert:
        figsize = (figsize[1], figsize[0])
    keys = sorted(stats.keys())
    intervals = []
    labels = []

    colors_map = {"Period": 'coral',"Interval": 'seagreen', "BD": 'royalblue',  "Delay": 'brown', "IBI": 'seagreen'}

    colors = []
    period_plot = False

    for key in keys:
        elem = stats[key]
        for e in reversed(sorted(elem.keys())):
            if(period_plot and e == "Period"):
                pass
            elif(e not in ignored_intervals):
                if e == "Period":
                    period_plot = True
                labels.append(key[1:] + "-" + e)
                # if(norm): # normalized data version ***beta***
                #     if(pos):
                #         # interval = np.absolute(stats[key][e])
                #         intervals.append(np.absolute(stats[key][e])/np.linalg.norm(stats[key][e]))
                #         # plt.plot()
                #         # plt.show()
                #     else:
                #         intervals.append(stats[key][e]/np.linalg.norm(stats[key][e]))
                # else:
                intervals.append(stats[key][e])

                colors.append(colors_map[e])
    
    plt.figure(figsize=figsize)
    bp = plt.boxplot(intervals, showfliers=False, labels=labels, patch_artist=True, vert=vert)

    # get legend grouping patches by color
    used = []
    legends = []
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
        if(color not in used):
            used.append(color)
            legends.append([patch, list(colors_map.keys())[list(colors_map.values()).index(color)]])

    for patch in bp['medians']:
        plt.setp(patch, color='black', linewidth=1.5)

    legends = np.array(legends)

    plt.tick_params(axis='both', labelsize=23)
    if vert:
        plt.xticks(rotation=45, ha='right')

    if box_ran is not None:
        if vert:
            plt.ylim(box_ran)
        else:
            plt.xlim(box_ran)

    if vert:
        plt.ylabel("Time intervals (ms)", fontsize=20)
    else:
        plt.xlabel("Time intervals (ms)", fontsize=20)

    plt.legend(legends[:,0],legends[:,1], fontsize='x-large', loc='best', bbox_to_anchor=(0.75,1))

    if title is None:
        plt.suptitle("Variability distribution for %d cycles"% len(stats[keys[0]]['Period']))
    else:
        plt.suptitle(title)

    plt.tight_layout()