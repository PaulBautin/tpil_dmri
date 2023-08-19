
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
import glob

import pandas as pd
import numpy as np
import os
import argparse
from functions.tpil_load_data import load_data_xlsx, diff_metrics, load_data_xlsx_add
from sklearn.cross_decomposition import PLSRegression, PLSSVD
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats
from scipy.sparse.linalg import svds
import pyls



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
        "-control",
        required=True,
        default='tractometry_results',
        help='Path to folder that contains output .xlsx files (e.g. "22-07-13_tractometry_CLBP/Statistics")',
    )
    optional = parser.add_argument_group("\nOPTIONAL ARGUMENTS")
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



def main():
    """
    main function, gather stats and call plots
    """
    pd.options.display.width = 0

    ### Get parser elements
    parser = get_parser()
    arguments = parser.parse_args()
    path_clbp = os.path.abspath(os.path.expanduser(arguments.clbp))
    path_control = os.path.abspath(os.path.expanduser(arguments.control))
    path_results_clbp = os.path.abspath(os.path.expanduser("/home/pabaua/Desktop/stats_clbp"))
    path_results_con = os.path.abspath(os.path.expanduser("/home/pabaua/Desktop/stats_control"))
    path_output = os.path.abspath(arguments.o)

    ## CLBP
    df_clbp = pd.read_csv(path_clbp, sep=',')
    df_clbp = df_clbp.drop(df_clbp.filter(regex='2M', axis=1).columns, axis=1)
    df_clbp = df_clbp.drop(df_clbp.filter(regex='4M', axis=1).columns, axis=1)
    df_clbp = df_clbp.drop('Groupe', axis=1)
    df_clbp['ID'] = 'sub-pl' + df_clbp['ID'].str.split('PT_PL').str[-1]
    df_clbp = df_clbp.rename(columns={'ID':'subject'}).set_index('subject')
    df_metric_clbp = load_data_xlsx(os.path.join(path_results_clbp, r'mean_std_per_point_per_subject.xlsx'),
                                   group_name='clbp')
    df_metric_clbp = df_metric_clbp[(df_metric_clbp["tract"] == '27_223_L') & (df_metric_clbp["session"] == "v1")].set_index(['subject', 'point'])['FAt_metric_mean'].reset_index()
    df_metric_clbp = df_metric_clbp.pivot(index='subject', columns='point', values='FAt_metric_mean')


    ## Control
    df_con = pd.read_csv(path_control, sep=',')
    df_con = df_con.drop(df_con.filter(regex='2M', axis=1).columns, axis=1)
    df_con = df_con.drop(df_con.filter(regex='4M', axis=1).columns, axis=1)
    df_con = df_con.drop('Groupe', axis=1)
    df_con['ID'] = 'sub-pl' + df_con['ID'].str.split('PT_PL').str[-1]
    df_con = df_con.rename(columns={'ID': 'subject'}).set_index('subject')
    df_metric_con = load_data_xlsx(os.path.join(path_results_con, r'mean_std_per_point_per_subject.xlsx'),
                                    group_name='con')
    df_metric_con = \
    df_metric_con[(df_metric_con["tract"] == '27_223_L') & (df_metric_con["session"] == "v1")].set_index(
        ['subject', 'point'])['FAt_metric_mean'].reset_index()
    df_metric_con = df_metric_con.pivot(index='subject', columns='point', values='FAt_metric_mean')
    print(df_con)

    ##

    # Partial Least Square (PLS)
    # my_plsSVD = PLSSVD(n_components=2, scale=True).set_output(transform="pandas")
    # pls = my_plsSVD.fit(df_clbp, df_metric_clbp)
    # X_c, Y_c = pls.transform(df_clbp, df_metric_clbp)
    # result = X_c.T @ Y_c

    # Partial Least Square (PLS) method 2
    # for i in df_metric_clbp.columns:
    #     df_metric_clbp[i] = stats.zscore(df_metric_clbp[i])
    # for i in df_clbp.columns:
    #     df_clbp[i] = stats.zscore(df_clbp[i])
    # clbp_metrics = np.vstack((df_metric_clbp.values.T, df_clbp.values.T))
    # R_covariance = np.cov(clbp_metrics)
    # u, s, v = svds(R_covariance)
    # print(v @ v.T)

    # Partial Least Square (PLS) method 2
    # R = print(pyls.compute.xcorr(df_metric_clbp, df_clbp))
    # U, S, V = pyls.compute.svd(R)
    # print(S.shape)
    # bpls = pyls.behavioral_pls(df_metric_clbp.values, df_clbp.values, groups=27, n_cond=1)
    # print(bpls.x_weights)


    # Principal Component Analysis (PCA)
    # pca = PCA(n_components=2)
    # pca.fit(df_clbp)
    # X_c = pca.transform(df_clbp)
    # print(pca.components_)
    # print(pca.explained_variance_ratio_)










    df_control = pd.read_csv(path_control, sep=',')
    df_metric_con = load_data_xlsx(os.path.join(path_results_con, r'mean_std_per_point_per_subject.xlsx'),
                                   group_name='con')
    df_metric_con = df_metric_con[(df_metric_con["tract"] == '27_223_L') & (df_metric_con["session"] == "v1")]
    df_participants = pd.read_csv("/home/pabaua/dev_tpil/data/22-08-19_dMRI_CLBP_BIDS/participants.tsv", sep='\t')






if __name__ == "__main__":
    main()