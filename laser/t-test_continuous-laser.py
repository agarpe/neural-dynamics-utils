import sys

import pandas as pd

from scipy.stats import ttest_ind, ttest_rel, f_oneway, mannwhitneyu

import superpos_functions as laser_utils

# get test value.
def t_test_by_group(data, metrics):
    groups = df.groupby(['file'])
    file_pvalues = []
    for group_name, df_group in groups:
        pvalues = [group_name]
        for metric in metrics:
            pvalue = ttest_ind(df_group[metric][df_group['type'] == 'laser'],
                                      df_group[metric][df_group['type'] == 'control']).pvalue
            pvalues.append(format(pvalue,'e'))
        
        file_pvalues.append(pvalues)

    from tabulate import tabulate   
    # print(tabulate(pvalues, headers=metrics, tablefmt='latex'))
    print(tabulate(file_pvalues, headers=metrics))

def t_test(group1, group2, metrics):
    pvalues = []
    for metric in metrics:
        # print("Group1 var, Group2 var")
        # print(group1[metric].var(), group2[metric].var())

        # # Student's t-test: assuming different variance 
        # tresult = ttest_ind(group1[metric], group2[metric])
        # print(metric, tresult)

        # Welch's t-test: assuming different variance 
        # tresult = ttest_ind(group1[metric], group2[metric],equal_var=False)
        tresult = ttest_rel(group1[metric], group2[metric]) #pair test - the correct one
        # tresult = mannwhitneyu(group1[metric], group2[metric])

        # print(metric, tresult)

        pvalues.append(format(tresult.pvalue,'e'))

    from tabulate import tabulate   
    # print(tabulate(pvalues, headers=metrics, tablefmt='latex'))
    print(tabulate([pvalues], headers=metrics))

# read dataframe (generated in analyze_and_plot_general_bar.py)
_dir = sys.argv[1]

df = pd.read_pickle(_dir+'/df_all_waveforms_metrics.pkl')

# Normalize data
metrics = ['duration','depolarization slope','repolarization slope','amplitude']
df = laser_utils.normalize_by(df, 'control', metrics)

df = laser_utils.clean_spikes(df, metrics, z_error=0.1)

# df,_ = laser_utils.get_sample(df, 1)
# df,_ = laser_utils.get_sample(df, 20)

# get group
df_controls = df[df['type']=='control']
df_lasers = df[df['type']=='laser']
df_recovery = df[df['type']=='recovery']

df_controls = df_controls.groupby('file').mean()
df_lasers = df_lasers.groupby('file').mean()
df_recovery = df_recovery.groupby('file').mean()

t_test_by_group(df, metrics)


print("\n Control vs lasers")
t_test(df_controls, df_lasers, metrics)
print("\n Control vs recovery")
t_test(df_controls, df_recovery, metrics)
print("\n Recovery vs lasers")
t_test(df_recovery, df_lasers, metrics)
