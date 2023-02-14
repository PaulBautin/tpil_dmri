# -*- coding: utf-8 -*-

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def boxplot_intersubject_per_ses(df, metric='nufo', bundle="NAC_mPFC_L_27"):
    df = df.loc[df['tract'] == bundle]
    df_v1 = df.loc[df['session'] == 'v1']
    df_v2 = df.loc[df['session'] == 'v2']
    df_v3 = df.loc[df['session'] == 'v3']
    fig, axs = plt.subplots(ncols=3, sharex=True, sharey=True)
    sns.violinplot(data=df_v1, x="group_name", y=metric, ax=axs[0])
    axs[0].set_title("0 months", fontsize=18)
    axs[0].set_ylabel("average NuFO", fontsize=18)
    axs[0].set_xlabel("group name", fontsize=16)
    axs[0].set_xticklabels(['control', 'CLBP'], fontsize=16)
    sns.violinplot(data=df_v2, x="group_name", y=metric, ax=axs[1])
    axs[1].set_title("2 months", fontsize=18)
    axs[1].set_ylabel("average NuFO", fontsize=18)
    axs[1].set_xlabel("group name", fontsize=16)
    axs[1].set_xticklabels(['control', 'CLBP'], fontsize=16)
    sns.violinplot(data=df_v3, x="group_name", y=metric, ax=axs[2])
    axs[2].set_title('4 months', fontsize=18)
    axs[2].set_ylabel("average NuFO", fontsize=18)
    axs[2].set_xlabel("group name", fontsize=16)
    axs[2].set_xticklabels(['control', 'CLBP'], fontsize=16)
    #fig.suptitle("Boxplot of " + metric + " for dcl and control subjects for each time point " + bundle, fontsize=12)
    sns.set(font_scale=2)
    #sns.violinplot(data=df, x="group_name", y=metric, ax=axs[3])
    #axs[3].set_title('total')
    plt.show()

def boxplot_intersubject(df, metric='nufo'):
    sns.violinplot(data=df, x="group_name", y=metric, palette="CMRmap_r").set(title="Boxplot of "+ metric + " for dcl and control subjects")
    plt.show()


def lineplot_t_test(df, metric='nufo', bundle="NAC_mPFC_L_27"):
    df = df.loc[df['tract'] == bundle]
    plt.title("p value of T-test between dcl and control subjects across metrics and sessions")
    plt.xlabel("position on bundle ")
    plt.ylabel("-log(p)")
    sns.lineplot(x=[0, len(df)/3], y=[-np.log(0.05), -np.log(0.05)], linestyle='--', color="black")
    df[metric] = df[metric].apply(lambda x: -np.log(x))
    sns.lineplot(data=df, x='point', y=metric, hue='session')
    plt.show()


def lineplot_per_point(df, metric='nufo', bundle="NAC_mPFC_L_27"):
    df = df.loc[df['tract'] == bundle]
    df_v1 = df.loc[df['session'] == 'v1']
    df_v2 = df.loc[df['session'] == 'v2']
    df_v3 = df.loc[df['session'] == 'v3']
    fig, axs = plt.subplots(nrows=3, sharex=True, sharey=True)
    sns.lineplot(data=df_v1, x='point', y=metric, hue='group_name', ax=axs[0])
    axs[0].set_title("0 months", fontsize=14)
    axs[0].set_ylabel("average NuFO", fontsize=14)
    h, l = axs[0].get_legend_handles_labels()
    axs[0].legend(handles=h, title='group name', labels=['control', 'CLBP'])
    sns.lineplot(data=df_v2, x='point', y=metric, hue='group_name', ax=axs[1])
    axs[1].set_title('2 months', fontsize=14)
    axs[1].set_ylabel("average NuFO", fontsize=14)
    h, l = axs[0].get_legend_handles_labels()
    axs[1].legend(handles=h, title='group name', labels=['control', 'CLBP'])
    sns.lineplot(data=df_v3, x='point', y=metric, hue='group_name', ax=axs[2])
    axs[2].set_title('4 months', fontsize=14)
    axs[2].set_ylabel("average NuFO", fontsize=14)
    axs[2].set_xlabel("position on bundle", fontsize=14)
    h, l = axs[0].get_legend_handles_labels()
    axs[2].legend(handles=h, title='group name', labels=['control', 'CLBP'])
    #fig.set_xlabel("position on bundle ")
    #sns.lineplot(data=df, x='point', y=metric, hue='group_name', ax=axs[3])
    #axs[3].set_title('total')
    #axs[3].set_xlabel("position on bundle ")
    fig.suptitle(metric + " for dcl and control subjects along the bundle " + bundle, fontsize=12)
    sns.set(font_scale=2)
    plt.show()


def lineplot_per_point_diff(df, metric='nufo', bundle="NAC_mPFC_L_27"):
    df = df.loc[df['tract'] == bundle]
    df_v1 = df.loc[df['session'] == 'v1']
    df_v2 = df.loc[df['session'] == 'v2']
    df_v3 = df.loc[df['session'] == 'v3']
    fig, axs = plt.subplots(nrows=3, sharex=True)
    sns.lineplot(data=df_v1, x='point', y=metric, ax=axs[0])
    axs[0].set_title(metric + " for dcl and control subjects along the bundle\n v1")
    sns.lineplot(data=df_v2, x='point', y=metric, ax=axs[1])
    axs[1].set_title('v2')
    sns.lineplot(data=df_v3, x='point', y=metric, ax=axs[2])
    axs[2].set_title('v3')
    axs[2].set_xlabel("position on bundle ")
    plt.show()


def heatmap_per_point(df, bundle="NAC_mPFC_L_27"):
    df = df.loc[df['tract'] == bundle]
    df = df[df.columns.drop(list(df.filter(regex='length')))]
    df = df[df.columns.drop(list(df.filter(regex='count')))]
    df = df[df.columns.drop(list(df.filter(regex='PCA')))]
    df_con = df.loc[df['group_name'] == 'con']
    df_clbp = df.loc[df['group_name'] == 'clbp']
    df_z = (df_clbp.groupby('point').mean() - df_con.groupby('point').mean()) / df_con.groupby('subject').mean().groupby('point').std()
    #ax = sns.heatmap(df_z.transpose(), annot=True)
    #plt.show()
    ax = sns.clustermap(df_z.transpose(), col_cluster=False, annot=True, cbar_pos=(0.90, 0.1, 0.02, 0.6))
    plt.show()
