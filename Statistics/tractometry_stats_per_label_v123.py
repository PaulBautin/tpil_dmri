
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

from functions.load_data import load_data_xlsx, df_gather_metrics



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





def main():
    """
    main function, gather stats and call plots
    """
    pd.options.display.width = 0
    ### Get parser elements
    #parser = get_parser()
    #arguments = parser.parse_args()
    #path_results_clbp = os.path.abspath(os.path.expanduser(arguments.clbp))
    # path_results_con = os.path.abspath(os.path.expanduser(arguments.con))
    #path_output = os.path.abspath(arguments.o)

    ### Form main Dataframes df_metrics_con for control subjects and df_metric_clbp for CLBP
    ## CLBP subjects
    df_mean_std_clbp = load_data_xlsx('/home/pabaua/Desktop/Statistics/mean_std_per_point_per_subject.xlsx')
    df_metrics_clbp = df_gather_metrics(df_mean_std_clbp, group_name="control", axis_name=["subject", "tract"])
    print(df_metrics_clbp)


if __name__ == "__main__":
    main()