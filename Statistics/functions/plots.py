# -*- coding: utf-8 -*-

import seaborn as sns
import matplotlib.pyplot as plt


def boxplot_intersubject_per_ses(df):
    sns.violinplot(data=df, x="session", y="ad_metric_mean", palette="CMRmap_r")
    plt.show()


def boxplot_per_point(df):
    df_metrics = df.set_index('tracts').loc["OR_ML_R"].reset_index()
    sns.set_theme(style="darkgrid")
    sns.lineplot(data=df_metrics, x="points", y=["md_metric_mean"], hue="group_name")
    plt.show()



