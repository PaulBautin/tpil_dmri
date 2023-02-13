
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
    files_subco_clbp = glob.glob(str(path_roi_clbp) + "/sub-*/subco_volume.txt")
    files_thick_clbp = glob.glob(str(path_roi_clbp) + "/sub-*/lh.thickness.txt")
    files_vol_clbp = glob.glob(str(path_roi_clbp) + "/sub-*/lh.volume.txt")
    files_subco_control = glob.glob(str(path_roi_control) + "/sub-*/subco_volume.txt")
    files_thick_control = glob.glob(str(path_roi_control) + "/sub-*/lh.thickness.txt")
    files_vol_control = glob.glob(str(path_roi_control) + "/sub-*/lh.volume.txt")
    df_participants = pd.read_csv("/home/pabaua/dev_tpil/data/22-08-19_dMRI_CLBP_BIDS/participants.tsv", sep='\t')
    df_participants['participant_id'] = df_participants['participant_id'].str.replace('-','-pl')
    path_output = os.path.abspath(arguments.o)

    # Get all data files and read json files into a dataframe
    df_roi_data_clbp = pd.concat(pd.read_csv(files_subco_clbp[i], sep='\t') for i in range(len(files_subco_clbp)))
    df_roi_data_clbp[['participant_id', 'session']] = df_roi_data_clbp['Measure:volume'].str.rsplit('_ses-', 1, expand=True)
    df_roi_clbp = df_roi_data_clbp.merge(df_participants, on="participant_id")
    df_roi_data_control = pd.concat(pd.read_csv(files_subco_control[i], sep='\t') for i in range(len(files_subco_control)))
    df_roi_data_control['Measure:volume'] = df_roi_data_control['Measure:volume'].str.rsplit('_T1w', 0).str[0]
    df_roi_data_control[['participant_id', 'session']] = df_roi_data_control['Measure:volume'].str.rsplit('_ses-', 1, expand=True)
    df_roi_control = df_roi_data_control.merge(df_participants, on="participant_id")
    df_zscore = np.abs((df_roi_clbp.mean() - df_roi_control.mean()) / df_roi_control.std())
    print(df_zscore[df_zscore > 0.4])
    df_roi_data = pd.concat([df_roi_control, df_roi_clbp])
    sns.violinplot(y=df_roi_data['Left-Inf-Lat-Vent'], x=df_roi_data['session'], hue=df_roi_data['group'])
    plt.show()

    # Get all data files and read json files into a dataframe
    df_roi_data_clbp = pd.concat(pd.read_csv(files_vol_clbp[i], sep='\t') for i in range(len(files_vol_clbp)))
    df_roi_data_clbp[['participant_id', 'session']] = df_roi_data_clbp['lh.BN_Atlas.volume'].str.rsplit('_ses-', 1,
                                                                                                    expand=True)
    df_roi_clbp = df_roi_data_clbp.merge(df_participants, on="participant_id")
    df_roi_data_control = pd.concat(pd.read_csv(files_vol_control[i], sep='\t') for i in range(len(files_vol_control)))
    df_roi_data_control['lh.BN_Atlas.volume'] = df_roi_data_control['lh.BN_Atlas.volume'].str.rsplit('_T1w', 0).str[0]
    df_roi_data_control[['participant_id', 'session']] = df_roi_data_control['lh.BN_Atlas.volume'].str.rsplit('_ses-', 1,
                                                                                                          expand=True)
    df_roi_control = df_roi_data_control.merge(df_participants, on="participant_id")
    df_zscore = np.abs((df_roi_clbp.mean() - df_roi_control.mean()) / df_roi_control.std())
    print(df_zscore[df_zscore > 0.5])
    df_roi_data = pd.concat([df_roi_control, df_roi_clbp])
    sns.violinplot(y=df_roi_data['lh_A22r_L_volume'], x=df_roi_data['session'], hue=df_roi_data['group'])
    plt.show()

if __name__ == "__main__":
    main()