
from __future__ import division

# !/usr/bin/env python
# -*- coding: utf-8
#########################################################################################
#
# Present statistics on control vs. chronic pain patients
#
# example: python tractometry_stat.py -i <results>
# ---------------------------------------------------------------------------------------
# Authors: Paul Bautin
#
# About the license: see the file LICENSE
#########################################################################################


import json

import pandas as pd
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from math import ceil
# from ruamel.yaml import YAML

#import plotly.express as px
from scipy import stats
import seaborn as sns



# Parser
#########################################################################################

def get_parser():
    """parser function"""
    parser = argparse.ArgumentParser(
        description="Compute statistics based on the .xlsx files containing the tractometry metrics:",
        formatter_class=argparse.RawTextHelpFormatter,
        prog=os.path.basename(__file__).strip(".py")
    )

    mandatory = parser.add_argument_group("\nMANDATORY ARGUMENTS")
    mandatory.add_argument(
        "-clbp",
        required=True,
        default='tractometry_results',
        help='Path to folder that contains output .xlsx files (e.g. "22-07-13_tractometry_CLBP/Statistics")',
    )
    mandatory.add_argument(
        "-con",
        required=True,
        default='tractometry_results',
        help='Path to folder that contains output .xlsx files (e.g. "22-07-13_tractometry_CON/Statistics")',
    )
    optional = parser.add_argument_group("\nOPTIONAL ARGUMENTS")
    optional.add_argument(
        '-config',
        required=False,
        help='Path to config file, which contains parameters for the statistics and figures. Example: config_script.yml',
    )
    optional.add_argument(
        '-fig',
        help='Generate figures',
        action='store_true'
    )
    optional.add_argument(
        '-o',
        help='Path where figures will be saved. By default, they will be saved in the current directory.',
        default="."
    )
    return parser


def load_data(file):
    """
    Creates a dataframe based on information present on TractometryFlow output xlsx files
    :param file: xlsx file containing information for statistics. Ex: 'mean_std.xlsx'
    :return sheet_to_df_map: 3D dataframe containing information present in each sheet of xlsx file
    """
    xls = pd.ExcelFile(file)
    sheet_to_df_map = {}
    for sheet_name in xls.sheet_names:
        sheet_to_df_map[sheet_name] = xls.parse(sheet_name)
        sheet_to_df_map[sheet_name].columns = ['ID']+[str(col) for col in sheet_to_df_map[sheet_name].columns[1:]]
    return sheet_to_df_map


def df_gather_metrics(df, group_name, axis_name):
    """
    Load and gather data from the ouput of 'load_data' following the tidy convention.
    :param df: 'load_data' output dataframe. Ex: df_mean_std_con
    :param group_name: identification of dataset group
    :param axis_name: New index naming for tidy format must be a (1x2) array. Ex: [ID, unnamed] -> [subject, tract]
    :return df_all: dataframe containing information present in xlsx file
    """
    df_all = pd.DataFrame()
    for metric in df.keys():
        df[metric] = df[metric].set_index("ID")
        df[metric] = df[metric].stack(dropna=False).rename(metric)
        df_all = pd.concat([df_all, df[metric]], axis=1)
    df_all = df_all.rename_axis(axis_name)
    df_all = df_all.reset_index()
    df_all["group_name"] = group_name
    return df_all


def diff_metrics(df_con, df_clbp, labels=None, diff=None):
    # prepare dataframe for PCA computation
    if diff is None:
        df = pd.concat([df_con, df_clbp]).dropna().groupby(labels).mean()
    if diff == "per_tract":
        df_con_copy = df_con.groupby(["tract"]).sample(n=1000, replace=True).drop(
            ["subject", "group_name", "point", "session"], axis=1).set_index("tract")
        df_clbp_copy = df_clbp.groupby(["tract"]).sample(n=1000, replace=True).drop(
            ["subject", "group_name", "point", "session"], axis=1).set_index("tract")
        df = df_con_copy.subtract(df_clbp_copy).dropna()
    if diff == "per_point":
        df_con_copy = df_con.groupby(["tract", "point"]).sample(n=1000, replace=True).drop(
            ["subject", "group_name", "session"], axis=1).set_index(["tract", "point"])
        df_clbp_copy = df_clbp.groupby(["tract", "point"]).sample(n=1000, replace=True).drop(
            ["subject", "group_name", "session"], axis=1).set_index(["tract", "point"])
        df = df_con_copy.subtract(df_clbp_copy).dropna()
    return df


def compute_pca(df_con, df_clbp, labels=None, diff=None, output_metrics=['PCA_1', 'PCA_2']):
    """
    Compute PCA metrics
    :param df_con: Control group dataframe
    :param df_clbp: Chronic Lower Back Pain group dataframe
    :param labels: labels to index dataframe
    :param diff: Compute PCA metrics on raw data diff=None, on difference of metrics pert tract: diff="per_tract"
    or on difference of metrics per point: diff="per_point"
    :param output_metrics: Number and name of PCA metrics to output
    :return df_pca: dataframe containing pca metrics
    :return pca: output of sklearn.decomposition.PCA
    :return x_norm: dataframe of the normalized initial metrics
    """
    # Isolate and standardize metrics (remove mean and scale to unit variance)
    df = diff_metrics(df_con, df_clbp, labels, diff)
    metrics = [i for i in df.keys() if "mean" in i]
    x = df.loc[:, metrics].values
    x_norm = StandardScaler().fit_transform(x)
    df_x_norm = pd.DataFrame(data=x_norm, index=df.index, columns=metrics)
    # Compute PCA metrics on standardized metrics
    pca = PCA(n_components=2)
    pc_metrics = pca.fit_transform(x_norm)
    df_pca = pd.DataFrame(data=pc_metrics, index=df.index, columns=output_metrics).reset_index()
    print('\nExplained variation per principal component: PC1 = {} and PC2 = {}'.format(pca.explained_variance_ratio_[0], pca.explained_variance_ratio_[1]))
    print('Importance of each metric per principal component: \n {}'.format(pd.DataFrame(pca.components_, columns=metrics, index=[['PCA_1', 'PCA_2']])))
    return df_pca, pca, df_x_norm


def figure_pca(df_pca):
    plt.xlabel('Principal Component #1')
    plt.ylabel('Principal Component #2')
    plt.title("Principal Component Analysis of brain microstructure metrics")
    sns.scatterplot(data=df_pca, x="PCA_1", y="PCA_2", hue="group_name")
    plt.show()

"""
def t_test(df_con, df_dcl, path_output, labels, pca=False, per_tract=False):
    if per_tract is False:
        df_dcl = df_dcl.set_index(labels).drop(["subject", "group_name", "session"], axis=1)
        df_con = df_con.set_index(labels).drop(["subject", "group_name", "session"], axis=1)
    elif per_tract:
        df_dcl = df_dcl.groupby([labels, "subject"]).mean().reset_index().set_index(labels).drop(["subject"], axis=1)
        df_con = df_con.groupby([labels, "subject"]).mean().reset_index().set_index(labels).drop(["subject"], axis=1)
    metrics = [i for i in df_con.keys() if "std" not in i]
    df_results = pd.DataFrame(index=df_dcl.groupby(labels).count().index)
    for metric in metrics:
        for ind in df_results.index:
            metrics_dcl = df_dcl.loc[ind][metric].values
            metrics_con = df_con.loc[ind][metric].values
            p_value = stats.ttest_ind(metrics_dcl, metrics_con, equal_var=True, nan_policy='omit')[1]
            df_results.loc[ind, metric] = p_value
    return df_results
"""
def t_test(df_con, df_dcl, labels, per_tract=False):
    if per_tract is False:
        df_dcl = df_dcl.set_index(labels).sort_index()
        df_con = df_con.set_index(labels).sort_index()
    elif per_tract:
        df_dcl = df_dcl.groupby([labels, "subject"]).mean().reset_index().set_index(labels)
        df_con = df_con.groupby([labels, "subject"]).mean().reset_index().set_index(labels)
    metrics = [i for i in df_con.keys() if "mean" in i or "PCA" in i]
    df_results = pd.DataFrame(index=df_dcl.groupby(labels).count().index)
    for metric in metrics:
        for ind in df_results.index:
            metrics_dcl = df_dcl.loc[ind][metric].values
            metrics_con = df_con.loc[ind][metric].values
            p_value = stats.ttest_ind(metrics_dcl, metrics_con, equal_var=True, nan_policy='omit')[1]
            df_results.loc[ind, metric] = p_value
    return df_results


def lineplot_t_test_per_point(df_metrics, bundle="OR_ML_R"):
    df_metrics = df_metrics.set_index("tract").loc[bundle].set_index("points")
    plt.title("p value of T-test between dcl and control subjects across metrics")
    plt.xlabel("position on bundle " + bundle)
    plt.ylabel("-log(p)")
    sns.lineplot(x=[0, len(df_metrics)], y=[-np.log(0.05), -np.log(0.05)], linestyle='--', color="black")
    print(df_metrics.columns)
    for i in df_metrics.columns:
        if np.any(-np.log(df_metrics[i].values) > -np.log(0.05)):
            sns.lineplot(data=-np.log(df_metrics[i]), palette="CMRmap_r", dashes=False, label=i)
        else:
            sns.lineplot(data=-np.log(df_metrics[i]), color='grey', alpha=0.2)
    plt.show()


def lineplot_per_point(df_con, df_dcl, df_metrics, bundle="AC", metric='fa_metric_mean'):
    df_con = df_con.set_index("tract").loc[bundle].reset_index()
    df_dcl = df_dcl.set_index("tract").loc[bundle].reset_index()
    df_metrics = df_metrics.set_index("tract").loc[bundle].set_index("point")
    fig, axs = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [1, 3]})
    # ax0
    axs[0].set_title("Average NuFO metric for CLBP and control subjects along bundle")
    axs[0].set_ylabel(metric)
    sns.lineplot(data=df_con, x="point", y=metric, hue="group_name", palette="CMRmap_r", ax=axs[0])
    sns.lineplot(data=df_dcl, x="point", y=metric, hue="group_name", ax=axs[0])
    # ax1
    #sns.lineplot(data=df_dcl, x="points", y=metric, hue="group_name", ax=axs[1])
    # ax2
    axs[1].set_title("p_value of T-test between CLBP and control subjects along bundle")
    axs[1].set_ylabel("-log(p_value)")
    axs[1].set_xlabel("position on bundle " + bundle)
    axs[1].legend(title="metric")
    sns.lineplot(x=[0, len(df_metrics)], y=[-np.log(0.05), -np.log(0.05)], linestyle='--', color="black", ax=axs[1])
    axs[1].text(0,-np.log(0.05)+0.1,"p_value = 0.05")
    for i in df_metrics.columns:
        if np.any(-np.log(df_metrics[i].values) > -np.log(0.05)):
            sns.lineplot(data=-np.log(df_metrics[i]), palette="CMRmap_r", dashes=False, label=i, ax=axs[1])
        else:
            sns.lineplot(data=-np.log(df_metrics[i]), color='grey', alpha=0.2, ax=axs[1])
    axs[1].legend(title="metric")
    plt.show()



def boxplot_intersubject_per_tract(df_metrics_per_tract):
    df_metrics = df_metrics_per_tract.set_index('tracts').loc["ICP_L"].stack()
    sns.violinplot(data=df_metrics, x="", y="ad_metric_mean", hue="group_name", palette="CMRmap_r", split=True)
    plt.show()


def boxplot_per_point(df_metrics):
    df_metrics = df_metrics.set_index('tracts').loc["OR_ML_R"].reset_index()
    sns.set_theme(style="darkgrid")
    sns.lineplot(data=df_metrics, x="points", y=["md_metric_mean"], hue="group_name")
    plt.show()


def figures(metrics_mean):
    # Figures
    fig = px.box(metrics_mean.set_index("tracts").loc["AC"].reset_index(), x="tracts", y="NUFO", title="figure DCL",
                 points="all", hover_data=["subjects"], color="kind")
    fig.show()
    # Correlation matrix
    corrMatrix = metrics_mean.corr()
    fig = px.imshow(corrMatrix, text_auto=True)
    fig.show()



def main():
    """
    main function, gather stats and call plots
    """
    pd.options.display.width = 0

    ### Get parser elements
    parser = get_parser()
    arguments = parser.parse_args()
    path_results_clbp = os.path.abspath(os.path.expanduser(arguments.clbp))
    path_results_con = os.path.abspath(os.path.expanduser(arguments.con))
    path_output = os.path.abspath(arguments.o)

    ### Form main Dataframes df_metrics_con for control subjects and df_metric_clbp for CLBP
    ## CON subjects
    df_mean_std_con = load_data(os.path.join(path_results_con, r'mean_std_per_label.xlsx'))
    df_metrics_con = df_gather_metrics(df_mean_std_con, group_name="control", axis_name=["subject", "tract"])
    df_metrics_con[['tract', 'point']] = df_metrics_con['tract'].str.rsplit('_', 1, expand=True)
    df_metrics_con[['subject', 'session']] = df_metrics_con['subject'].str.rsplit('_', 1, expand=True)
    ## CLBP subjects
    df_mean_std_clbp = load_data(os.path.join(path_results_clbp, r'mean_std_per_label.xlsx'))
    df_metrics_clbp = df_gather_metrics(df_mean_std_clbp, group_name="CLBP", axis_name=["subject", "tract"])
    df_metrics_clbp[['tract', 'point']] = df_metrics_clbp['tract'].str.rsplit('_', 1, expand=True)
    df_metrics_clbp[['subject', 'session']] = df_metrics_clbp['subject'].str.rsplit('_', 1, expand=True)

    ### Compute PCA metrics
    ## Compute PCA metrics per tract
    # df_pca_per_tract, pca_per_tract, df_x_norm_per_tract, = compute_pca(df_metrics_con, df_metrics_clbp, labels=["tracts", "subjects", "group_name"])
    # df_pca_per_tract_diff, pca_per_tract_diff, df_x_norm_per_tract_diff, = compute_pca(df_metrics_con, df_metrics_clbp, labels=["tracts"], diff="per_tract")
    # df_pca_per_tract_inv = pd.DataFrame(data=pca_per_tract_diff.transform(df_x_norm_per_tract.values), index=df_x_norm_per_tract.index, columns=df_pca_per_tract.columns)
    ## Compute PCA metrics per point
    print("\n ######### PCA per point ###########")
    df_pca_per_point, pca_per_point, df_x_norm_per_point, = compute_pca(df_metrics_con, df_metrics_clbp,
                                                                        labels=["tract", "subject", "point",
                                                                                "group_name", "session"])
    df_pca_per_point_con = df_pca_per_point[df_pca_per_point["group_name"] == "control"]
    df_pca_per_point_clbp = df_pca_per_point[df_pca_per_point["group_name"] == "CLBP"]
    ## Compute PCA per point of difference between control and DCL
    print("\n ######### PCA per point diff ###########")
    df_pca_per_point_diff, pca_per_point_diff, df_x_norm_per_point_diff, = compute_pca(df_metrics_con, df_metrics_clbp,
                                                                                       diff="per_point",
                                                                                       output_metrics=['PCA_1_diff', 'PCA_2_diff'])
    # Apply per point diff PCA on per point metrics
    df_pca_per_point_inv = pd.DataFrame(data=pca_per_point_diff.transform(df_x_norm_per_point.values),
                                        index=df_x_norm_per_point.index, columns=['PCA_1_diff', 'PCA_2_diff']).reset_index()
    df_pca_per_point_inv_con = df_pca_per_point_inv[df_pca_per_point_inv["group_name"] == "control"]
    df_pca_per_point_inv_clbp = df_pca_per_point_inv[df_pca_per_point_inv["group_name"] == "CLBP"]


    # Add metrics PCA and PCA_diff to main dataframe
    df_metrics_con = df_metrics_con.merge(df_pca_per_point_con, how="left",
                                          on=["tract", "subject", "point", "group_name", "session"])
    df_metrics_clbp = df_metrics_clbp.merge(df_pca_per_point_clbp, how="left",
                                          on=["tract", "subject", "point", "group_name", "session"])
    df_metrics_con = df_metrics_con.merge(df_pca_per_point_inv_con, how="left",
                                          on=["tract", "subject", "point", "group_name", "session"])
    df_metrics_clbp = df_metrics_clbp.merge(df_pca_per_point_inv_clbp, how="left",
                                          on=["tract", "subject", "point", "group_name", "session"])


    ### Stats
    ## Stats per point
    # t_test between Control and DCL
    df_t_test_per_point = t_test(df_metrics_con, df_metrics_clbp, labels=["tract", "point"])
    per_point_results = df_t_test_per_point.stack()[df_t_test_per_point.stack() < (0.01)].dropna().unstack()
    # per_point_results.to_csv(os.path.join(path_output, "per_point_results.csv"))
    #print('[')
    #for val in df_t_test_per_point.loc["AF_L"]["fa_metric_mean"].values:
    #    print('{},'.format(val))
    #print(']')
    ## Stats per tract
    # t_test between Control and DCL
    df_t_test_per_tract = t_test(df_metrics_con, df_metrics_clbp, labels="tract", per_tract=True)
    per_tract_results = df_t_test_per_tract.stack()[df_t_test_per_tract.stack() < (0.1)].dropna().unstack()
    # per_tract_results.to_csv(os.path.join(path_output, "per_tract_results.csv"))

    ## Figures
    if arguments.fig:
        df_metrics_con.to_csv(path_output + "/df_metrics_con.csv")
        df_metrics_clbp.to_csv(path_output + "/df_metrics_clbp.csv")
        df_t_test_per_point.to_csv(path_output + "/df_t_test_per_point.csv")
        lineplot_per_point(pd.read_csv(path_output + "/df_metrics_con.csv"),
                           pd.read_csv(path_output + "/df_metrics_clbp.csv"),
                           pd.read_csv(path_output + "/df_t_test_per_point.csv"),
                           bundle="NAC_mPFC_L", metric='nufo_metric_mean')
    #lineplot_t_test_per_point(pd.read_csv(path_output + "/df_t_test_per_point.csv"),
    #                          bundle="AC")
    # boxplot per tract of subject
    #boxplot_intersubject_per_tract(df_metrics_per_tract)
    # boxplot_per_point(df_metrics_per_point)
    #

if __name__ == "__main__":
    main()