
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

from functions.load_data import load_data_xlsx, diff_metrics
from functions.plots import boxplot_intersubject_per_ses, lineplot_t_test, lineplot_per_point, lineplot_per_point_diff, boxplot_intersubject
from functions.stat_tests import t_test_longitudinal, t_test_cs, t_test_cs_per_session, t_test_cs_per_session_per_point, t_test_cs_mean
from functions.pca import fit_pca, apply_pca



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
    path_results_con = os.path.abspath(os.path.expanduser(arguments.con))
    path_results_clbp = os.path.abspath(os.path.expanduser(arguments.clbp))
    path_output = os.path.abspath(arguments.o)

    ### Form main Dataframes df_metrics_con for control subjects and df_metric_clbp for CLBP
    ## CON subjects
    df_metric_con = load_data_xlsx(os.path.join(path_results_con, r'mean_std_per_point_per_subject.xlsx'), group_name='con')
    ## CLBP subjects
    df_metric_clbp = load_data_xlsx(os.path.join(path_results_clbp, r'mean_std_per_point_per_subject.xlsx'), group_name='clbp')
    ## Concatenate CON and CLBP subjects
    df_metric = pd.concat([df_metric_con, df_metric_clbp])
    ## Difference metrics between CON and CLBP
    df_diff_metric = diff_metrics(df_metric_con, df_metric_clbp)

    ### Compute PCA metrics
    ## CON + CLBP subjects
    pca, df_x_norm = fit_pca(df_metric, ['group_name', 'subject', 'session', 'tract', 'point'])
    df_pca = apply_pca(pca, df_x_norm)
    df_metric = df_metric.merge(df_pca, how="left", on=['group_name', 'subject', 'session', 'tract', 'point'])
    ## Difference metrics between CON and CLBP
    pca_diff, df_x_norm_diff = fit_pca(df_diff_metric, ["session", "tract", "point"])
    df_d_pca = apply_pca(pca_diff, df_x_norm_diff, output_metrics=['PCA_1', 'PCA_2'])
    df_diff_pca = apply_pca(pca_diff, df_x_norm, output_metrics=['PCA_1_diff', 'PCA_2_diff'])
    df_metric = df_metric.merge(df_diff_pca, how="left", on=['group_name', 'subject', 'session', 'tract', 'point'])
    df_diff_metric = df_diff_metric.merge(df_d_pca, how="left", on=['session', 'tract', 'point'])

    ### Stats
    ## Stats per point
    # t_test between Control and DCL
    df_t_test_cs = t_test_cs_per_session_per_point(df_metric)
    print(df_t_test_cs[df_t_test_cs < 0.05].dropna(how='all').dropna(axis=1, how='all'))
    #df_t_test_longitudinal = t_test_longitudinal(df_metric)
    #print(df_t_test_longitudinal[df_t_test_longitudinal < 0.05].dropna(how='all').dropna(axis=1, how='all'))


    ### Figures
    lineplot_per_point(df_metric, metric='nufo_metric_mean')
    #lineplot_per_point_diff(df_diff_metric, metric='PCA_1')
    #boxplot_intersubject(df_metric, metric='noddi_icvf_metric_mean')
    boxplot_intersubject_per_ses(df_metric, metric='nufo_metric_mean')

    #t_test_longitudinal(df_metrics_clbp)

    #pca, df_x_norm = fit_pca(df_metrics_clbp)
    #df_pca = apply_pca(pca, df_x_norm)

if __name__ == "__main__":
    main()