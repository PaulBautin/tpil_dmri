# -*- coding: utf-8 -*-

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def boxplot_intersubject_per_ses(df, metric='nufo'):
    fig, axs = plt.subplots(ncols=3)
    df_v1 = df.loc[df['session'] == 'v1']
    df_v2 = df.loc[df['session'] == 'v2']
    df_v3 = df.loc[df['session'] == 'v3']
    sns.violinplot(data=df_v1, x="group_name", y=metric, palette="CMRmap_r", ax=axs[0])
    axs[0].set_title("v1")
    sns.violinplot(data=df_v2, x="group_name", y=metric, palette="CMRmap_r", ax=axs[1])
    axs[1].set_title("Boxplot of "+ metric + " for dcl and control subjects for each time point\nv2")
    sns.violinplot(data=df_v3, x="group_name", y=metric, palette="CMRmap_r", ax=axs[2])
    axs[2].set_title('v3')
    plt.show()

def boxplot_intersubject(df, metric='nufo'):
    sns.violinplot(data=df, x="group_name", y=metric, palette="CMRmap_r").set(title="Boxplot of "+ metric + " for dcl and control subjects")
    plt.show()


def lineplot_t_test(df, metric='nufo'):
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
    axs[0].set_title(metric + " for dcl and control subjects along the bundle\n v1")
    sns.lineplot(data=df_v2, x='point', y=metric, hue='group_name', ax=axs[1])
    axs[1].set_title('v2')
    sns.lineplot(data=df_v3, x='point', y=metric, hue='group_name', ax=axs[2])
    axs[2].set_title('v3')
    axs[2].set_xlabel("position on bundle ")
    plt.show()


def lineplot_per_point_diff(df, metric='nufo'):
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



