
from __future__ import division

# !/usr/bin/env python
# -*- coding: utf-8
#########################################################################################
#
# Network control theory control vs. chronic pain patients
#
# example: python nct_main.py -i <results>
# ---------------------------------------------------------------------------------------
# Authors: Paul Bautin
#
# About the license: see the file LICENSE
#########################################################################################


import numpy as np
import pandas as pd
import glob
from os.path import dirname as up


def cortical_volumes_2(path_roi_clbp, path_roi_con):
    """
    Calculate the cortical volumes for each subject and session
    :param path_roi_clbp: path to CLBP freesurfer ROI volumes. Ex: '/home/pabaua/dev_tpil/data/Freesurfer/22-09-21_t1_clbp_freesurfer_output'
    :param path_roi_con: path to control freesurfer ROI volumes. Ex: '/home/pabaua/dev_tpil/data/Freesurfer/23_02_09_control_freesurfer_output'
    """

    # Get the freesurfer files
    files_clbp_lh = glob.glob(str(path_roi_clbp) + "/sub-*/lh.volume.txt")
    files_clbp_rh = glob.glob(str(path_roi_clbp) + "/sub-*/rh.volume.txt")
    files_control_lh = glob.glob(str(path_roi_control) + "/sub-*/lh.volume.txt")
    files_control_rh = glob.glob(str(path_roi_control) + "/sub-*/rh.volume.txt")

    ## clbp lh
    df_clbp_lh = pd.concat(pd.read_csv(files_clbp_lh[i], sep='\t') for i in range(len(files_clbp_lh)))
    df_clbp_lh[['participant_id', 'session']] = df_clbp_lh['lh.BN_Atlas.volume'].str.rsplit('_ses-', n=1, expand=True)
    df_clbp_lh_avg = df_clbp_lh.drop(["lh.BN_Atlas.volume", "participant_id"], axis=1).groupby(['session']).mean()

    ## control lh
    df_control_lh = pd.concat(
        pd.read_csv(files_control_lh[i], sep='\t') for i in range(len(files_control_lh)))
    df_control_lh['lh.BN_Atlas.volume'] = df_control_lh['lh.BN_Atlas.volume'].str.rsplit('_T1w', n=0).str[0]
    df_control_lh[['participant_id', 'session']] = df_control_lh['lh.BN_Atlas.volume'].str.rsplit('_ses-', n=1, expand=True)
    df_control_lh_avg = df_control_lh.drop(["lh.BN_Atlas.volume", "participant_id"], axis=1).groupby(['session']).mean()
    df_control_lh_std = df_control_lh.drop(["lh.BN_Atlas.volume"], axis=1).groupby(['session', 'participant_id']).mean().groupby(['session']).std()

    # diff and zscore lh
    df_lh_diff = (df_clbp_lh_avg - df_control_lh_avg) / df_control_lh_avg
    df_lh_z_score = (df_clbp_lh_avg - df_control_lh_avg) / df_control_lh_std

    ## clbp rh
    df_clbp_rh = pd.concat(pd.read_csv(files_clbp_rh[i], sep='\t') for i in range(len(files_clbp_rh)))
    df_clbp_rh[['participant_id', 'session']] = df_clbp_rh['rh.BN_Atlas.volume'].str.rsplit('_ses-', n=1, expand=True)
    df_clbp_rh_avg = df_clbp_rh.drop(["rh.BN_Atlas.volume", "participant_id"], axis=1).groupby(['session']).mean()

    ## control rh
    df_control_rh = pd.concat(pd.read_csv(files_control_rh[i], sep='\t') for i in range(len(files_control_rh)))
    df_control_rh['rh.BN_Atlas.volume'] = df_control_rh['rh.BN_Atlas.volume'].str.rsplit('_T1w', n=0).str[0]
    df_control_rh[['participant_id', 'session']] = df_control_rh['rh.BN_Atlas.volume'].str.rsplit('_ses-', n=1, expand=True)
    df_control_rh_avg = df_control_rh.drop(["rh.BN_Atlas.volume", "participant_id"], axis=1).groupby(['session']).mean()
    df_control_rh_std = df_control_rh.drop(["rh.BN_Atlas.volume"], axis=1).groupby(['session', 'participant_id']).mean().groupby(['session']).std()

    # diff and zscore rh
    df_rh_diff = (df_clbp_rh_avg - df_control_rh_avg) / df_control_rh_avg
    df_rh_z_score = (df_clbp_rh_avg - df_control_rh_avg) / df_control_rh_std

    diff = {ses: [val for pair in zip(df_lh_diff.loc[ses], df_rh_diff.loc[ses]) for val in pair][:210] for ses in ['v1', 'v2', 'v3']}
    z_score = {ses: [val for pair in zip(df_lh_z_score.loc[ses], df_rh_z_score.loc[ses]) for val in pair][:210] for ses in ['v1', 'v2', 'v3']}
    return diff, z_score


def cortical_measures(path_roi, atlas='BN_Atlas.volume'):
    """
    Calculate the cortical volumes for each subject and session
    :param path_roi: path to freesurfer ROI volumes. Ex: '/home/pabaua/dev_tpil/data/Freesurfer/22-09-21_t1_clbp_freesurfer_output'
    """

    # Get the freesurfer files
    files_lh = glob.glob(str(path_roi) + "/sub-*/lh.volume.txt")
    files_rh = glob.glob(str(path_roi) + "/sub-*/rh.volume.txt")
    print(files_lh)

    ## Create dataframe for lh
    df_lh = pd.concat(pd.read_csv(files_clbp_lh[i], sep='\t') for i in range(len(files_clbp_lh)))
    df_lh[['participant_id', 'session']] = df_clbp_lh['lh.'+ atlas].str.rsplit('_ses-', n=1, expand=True)
    df_lh_avg = df_clbp_lh.drop(['lh.'+ atlas, "participant_id"], axis=1).groupby(['session']).mean()
    print(df_lh_avg)

    ## Create dataframe for rh
    df_rh = pd.concat(pd.read_csv(files_clbp_rh[i], sep='\t') for i in range(len(files_clbp_rh)))
    df_rh[['participant_id', 'session']] = df_clbp_rh['rh.'+ atlas].str.rsplit('_ses-', n=1, expand=True)
    df_rh_avg = df_clbp_rh.drop(['rh.'+ atlas, "participant_id"], axis=1).groupby(['session']).mean()
    return diff, z_score

