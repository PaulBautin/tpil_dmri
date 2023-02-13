
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
import json
import matplotlib.pyplot as plt
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
    path_roi_clbp = os.path.abspath(os.path.expanduser(arguments.clbp))
    path_roi_control = os.path.abspath(os.path.expanduser(arguments.control))
    files_clbp = glob.glob(str(path_roi_clbp) + "/sub-*/subco_volume.txt")
    files_control = glob.glob(str(path_roi_control) + "/sub-*/subco_volume.txt")
    df_participants = pd.read_csv("/home/pabaua/dev_tpil/data/22-08-19_dMRI_CLBP_BIDS/participants.tsv", sep='\t')
    path_output = os.path.abspath(arguments.o)

    # Get all data files and read json files into a dataframe
    df_roi_data_clbp = pd.concat(pd.read_csv(files_clbp[i], sep='\t') for i in range(len(files_control))).set_index("Measure:volume")
    print(df_roi_data_clbp)
    df_roi_data_clbp['group_name'] = "clbp"
    df_roi_data_control = pd.concat(pd.read_csv(files_control[i], sep='\t') for i in range(len(files_control)))
    df_roi_data_control['group_name'] = "control"
    df_roi_data = pd.concat([df_roi_data_control, df_roi_data_clbp])
    sns.boxplot(y=df_roi_data['Left-Accumbens-area'], x=df_roi_data['group_name'])
    plt.show()
    # intersubject_data = pd.read_json(intersubject_json_files[0])
    # print("\n\n########### intersubject results: ###########\n{}".format(intersubject_data.mean()))
    #
    # intrasubject_json_files = glob(path_intra_clbp + "/*/27_223_Pairwise_Comparaison.json")
    # intrasubject_data = pd.concat([pd.read_json(intrasubject_json_files[i]).mean() for i in range(len(intrasubject_json_files))])
    # print("\n\n########### intrasubject results: ###########\n{}".format(intrasubject_data.groupby(intrasubject_data.index).mean()))

if __name__ == "__main__":
    main()