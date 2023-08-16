
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
from netneurotools.datasets import fetch_conte69
import nibabel as nib
from netneurotools.plotting import plot_point_brain
from netneurotools.utils import get_centroids
from enigmatoolbox.plotting import plot_subcortical


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


def cortical_volume_analysis(path_roi_clbp, path_roi_control, df_participants):
    files_vol_clbp_lh = glob.glob(str(path_roi_clbp) + "/sub-*/lh.volume.txt")
    files_vol_clbp_rh = glob.glob(str(path_roi_clbp) + "/sub-*/rh.volume.txt")
    files_vol_control_lh = glob.glob(str(path_roi_control) + "/sub-*/lh.volume.txt")
    files_vol_control_rh = glob.glob(str(path_roi_control) + "/sub-*/rh.volume.txt")
    df_participants['participant_id'] = df_participants['participant_id'].str.replace('-', '-pl')

    ## clbp lh
    df_vol_clbp_lh = pd.concat(pd.read_csv(files_vol_clbp_lh[i], sep='\t') for i in range(len(files_vol_clbp_lh)))
    df_vol_clbp_lh[['participant_id', 'session']] = df_vol_clbp_lh['lh.BN_Atlas.volume'].str.rsplit('_ses-', 1, expand=True)
    df_vol_clbp_lh = df_vol_clbp_lh.merge(df_participants, on="participant_id")
    df_vol_clbp_lh["group"] = "clbp"
    df_vol_clbp_lh["hemisphere"] = 'lh'
    df_vol_clbp_lh = df_vol_clbp_lh.drop(["lh.BN_Atlas.volume"], axis=1)

    ## control lh
    df_vol_control_lh = pd.concat(
        pd.read_csv(files_vol_control_lh[i], sep='\t') for i in range(len(files_vol_control_lh)))
    df_vol_control_lh['lh.BN_Atlas.volume'] = df_vol_control_lh['lh.BN_Atlas.volume'].str.rsplit('_T1w', 0).str[0]
    df_vol_control_lh[['participant_id', 'session']] = df_vol_control_lh['lh.BN_Atlas.volume'].str.rsplit('_ses-', 1,
                                                                                                          expand=True)
    df_vol_control_lh = df_vol_control_lh.merge(df_participants, on="participant_id")
    df_vol_control_lh["group"] = "control"
    df_vol_control_lh["hemisphere"] = 'lh'
    df_vol_control_lh = df_vol_control_lh.drop(["lh.BN_Atlas.volume"], axis=1)

    df_vol_lh = pd.concat([df_vol_control_lh, df_vol_clbp_lh])
    #df_vol_lh = df_vol_lh.groupby(['group', 'participant_id', 'age', 'session', 'hemisphere']).mean()




    ## clbp rh
    df_vol_clbp_rh = pd.concat(pd.read_csv(files_vol_clbp_rh[i], sep='\t') for i in range(len(files_vol_clbp_rh)))
    df_vol_clbp_rh[['participant_id', 'session']] = df_vol_clbp_rh['rh.BN_Atlas.volume'].str.rsplit('_ses-', 1, expand=True)
    df_vol_clbp_rh = df_vol_clbp_rh.merge(df_participants, on="participant_id")
    df_vol_clbp_rh["group"] = "clbp"
    df_vol_clbp_rh["hemisphere"] = 'rh'
    df_vol_clbp_rh = df_vol_clbp_rh.drop(["rh.BN_Atlas.volume"], axis=1)


    ## control rh
    df_vol_control_rh = pd.concat(pd.read_csv(files_vol_control_rh[i], sep='\t') for i in range(len(files_vol_control_rh)))
    df_vol_control_rh['rh.BN_Atlas.volume'] = df_vol_control_rh['rh.BN_Atlas.volume'].str.rsplit('_T1w', 0).str[0]
    df_vol_control_rh[['participant_id', 'session']] = df_vol_control_rh['rh.BN_Atlas.volume'].str.rsplit('_ses-', 1, expand=True)
    df_vol_control_rh = df_vol_control_rh.merge(df_participants, on="participant_id")
    df_vol_control_rh["group"] = "control"
    df_vol_control_rh["hemisphere"] = 'rh'
    df_vol_control_rh = df_vol_control_rh.drop(["rh.BN_Atlas.volume"], axis=1)

    df_vol_rh = pd.concat([df_vol_control_rh, df_vol_clbp_rh])


    #df_vol_rh = df_vol_rh.groupby(['group', 'participant_id', 'age', 'session', 'hemisphere']).mean()
    df_vol_rh = df_vol_rh.drop(['BrainSegVolNotVent', 'eTIV'], axis=1)
    df_vol_lh = df_vol_lh.drop(['BrainSegVolNotVent', 'eTIV'], axis=1)
    df_vol_lh = df_vol_lh.set_index(['group', 'participant_id', 'age', 'sex', 'session', 'hemisphere'])
    df_vol_lh.columns = df_vol_lh.columns.str.split('lh_').str[1]
    df_vol_lh.columns = df_vol_lh.columns.str.split('_L_volume').str[0]
    df_vol_rh = df_vol_rh.set_index(['group', 'participant_id', 'age', 'sex', 'session', 'hemisphere'])
    df_vol_rh.columns = df_vol_rh.columns.str.split('rh_').str[1]
    df_vol_rh.columns = df_vol_rh.columns.str.split('_R_volume').str[0]
    df_vol = pd.concat([df_vol_lh, df_vol_rh])
    df_vol = df_vol.reset_index().groupby(['group', 'participant_id', 'age', 'sex', 'session', 'hemisphere']).mean()
    return df_vol

def cortical_thick_analysis(path_roi_clbp, path_roi_control, df_participants):
    files_clbp_lh = glob.glob(str(path_roi_clbp) + "/sub-*/lh.thickness.txt")
    files_clbp_rh = glob.glob(str(path_roi_clbp) + "/sub-*/rh.thickness.txt")
    files_control_lh = glob.glob(str(path_roi_control) + "/sub-*/lh.thickness.txt")
    files_control_rh = glob.glob(str(path_roi_control) + "/sub-*/rh.thickness.txt")
    #df_participants['participant_id'] = df_participants['participant_id'].str.replace('-', '-pl')

    ## clbp lh
    df_clbp_lh = pd.concat(pd.read_csv(files_clbp_lh[i], sep='\t') for i in range(len(files_clbp_lh)))
    df_clbp_lh[['participant_id', 'session']] = df_clbp_lh['lh.BN_Atlas.thickness'].str.rsplit('_ses-', 1, expand=True)
    df_clbp_lh = df_clbp_lh.merge(df_participants, on="participant_id")
    df_clbp_lh["group"] = "clbp"
    df_clbp_lh["hemisphere"] = 'lh'
    df_clbp_lh = df_clbp_lh.drop(["lh.BN_Atlas.thickness"], axis=1)


    ## control lh
    df_control_lh = pd.concat(
        pd.read_csv(files_control_lh[i], sep='\t') for i in range(len(files_control_lh)))
    df_control_lh['lh.BN_Atlas.thickness'] = df_control_lh['lh.BN_Atlas.thickness'].str.rsplit('_T1w', 0).str[0]
    df_control_lh[['participant_id', 'session']] = df_control_lh['lh.BN_Atlas.thickness'].str.rsplit('_ses-', 1,
                                                                                                          expand=True)
    df_control_lh = df_control_lh.merge(df_participants, on="participant_id")
    df_control_lh["group"] = "control"
    df_control_lh["hemisphere"] = 'lh'
    df_control_lh = df_control_lh.drop(["lh.BN_Atlas.thickness"], axis=1)

    df_lh = pd.concat([df_control_lh, df_clbp_lh])

    ## clbp rh
    df_clbp_rh = pd.concat(pd.read_csv(files_clbp_rh[i], sep='\t') for i in range(len(files_clbp_rh)))
    df_clbp_rh[['participant_id', 'session']] = df_clbp_rh['rh.BN_Atlas.thickness'].str.rsplit('_ses-', 1, expand=True)
    df_clbp_rh = df_clbp_rh.merge(df_participants, on="participant_id")
    df_clbp_rh["group"] = "clbp"
    df_clbp_rh["hemisphere"] = 'rh'
    df_clbp_rh = df_clbp_rh.drop(["rh.BN_Atlas.thickness"], axis=1)


    ## control rh
    df_control_rh = pd.concat(pd.read_csv(files_control_rh[i], sep='\t') for i in range(len(files_control_rh)))
    df_control_rh['rh.BN_Atlas.thickness'] = df_control_rh['rh.BN_Atlas.thickness'].str.rsplit('_T1w', 0).str[0]
    df_control_rh[['participant_id', 'session']] = df_control_rh['rh.BN_Atlas.thickness'].str.rsplit('_ses-', 1, expand=True)
    df_control_rh = df_control_rh.merge(df_participants, on="participant_id")
    df_control_rh["group"] = "control"
    df_control_rh["hemisphere"] = 'rh'
    df_control_rh = df_control_rh.drop(["rh.BN_Atlas.thickness"], axis=1)

    df_rh = pd.concat([df_control_rh, df_clbp_rh])

    return df_lh, df_rh

def cortical_volume_analysis_2(path_roi_clbp, path_roi_control, df_participants):
    files_clbp_lh = glob.glob(str(path_roi_clbp) + "/sub-*/lh.volume.txt")
    files_clbp_rh = glob.glob(str(path_roi_clbp) + "/sub-*/rh.volume.txt")
    files_control_lh = glob.glob(str(path_roi_control) + "/sub-*/lh.volume.txt")
    files_control_rh = glob.glob(str(path_roi_control) + "/sub-*/rh.volume.txt")
    df_participants['participant_id'] = df_participants['participant_id'].str.replace('-', '-pl')

    ## clbp lh
    df_clbp_lh = pd.concat(pd.read_csv(files_clbp_lh[i], sep='\t') for i in range(len(files_clbp_lh)))
    df_clbp_lh[['participant_id', 'session']] = df_clbp_lh['lh.BN_Atlas.volume'].str.rsplit('_ses-', 1, expand=True)
    df_clbp_lh = df_clbp_lh.merge(df_participants, on="participant_id")
    df_clbp_lh["group"] = "clbp"
    df_clbp_lh["hemisphere"] = 'lh'
    df_clbp_lh = df_clbp_lh.drop(["lh.BN_Atlas.volume"], axis=1)


    ## control lh
    df_control_lh = pd.concat(
        pd.read_csv(files_control_lh[i], sep='\t') for i in range(len(files_control_lh)))
    df_control_lh['lh.BN_Atlas.volume'] = df_control_lh['lh.BN_Atlas.volume'].str.rsplit('_T1w', 0).str[0]
    df_control_lh[['participant_id', 'session']] = df_control_lh['lh.BN_Atlas.volume'].str.rsplit('_ses-', 1,
                                                                                                          expand=True)
    df_control_lh = df_control_lh.merge(df_participants, on="participant_id")
    df_control_lh["group"] = "control"
    df_control_lh["hemisphere"] = 'lh'
    df_control_lh = df_control_lh.drop(["lh.BN_Atlas.volume"], axis=1)

    df_lh = pd.concat([df_control_lh, df_clbp_lh])


    ## clbp rh
    df_clbp_rh = pd.concat(pd.read_csv(files_clbp_rh[i], sep='\t') for i in range(len(files_clbp_rh)))
    df_clbp_rh[['participant_id', 'session']] = df_clbp_rh['rh.BN_Atlas.volume'].str.rsplit('_ses-', 1, expand=True)
    df_clbp_rh = df_clbp_rh.merge(df_participants, on="participant_id")
    df_clbp_rh["group"] = "clbp"
    df_clbp_rh["hemisphere"] = 'rh'
    df_clbp_rh = df_clbp_rh.drop(["rh.BN_Atlas.volume"], axis=1)


    ## control rh
    df_control_rh = pd.concat(pd.read_csv(files_control_rh[i], sep='\t') for i in range(len(files_control_rh)))
    df_control_rh['rh.BN_Atlas.volume'] = df_control_rh['rh.BN_Atlas.volume'].str.rsplit('_T1w', 0).str[0]
    df_control_rh[['participant_id', 'session']] = df_control_rh['rh.BN_Atlas.volume'].str.rsplit('_ses-', 1, expand=True)
    df_control_rh = df_control_rh.merge(df_participants, on="participant_id")
    df_control_rh["group"] = "control"
    df_control_rh["hemisphere"] = 'rh'
    df_control_rh = df_control_rh.drop(["rh.BN_Atlas.volume"], axis=1)

    df_rh = pd.concat([df_control_rh, df_clbp_rh])

    return df_lh, df_rh

def subcortical_analysis(path_roi_clbp, path_roi_control, df_participants):
    files_subco_clbp = glob.glob(str(path_roi_clbp) + "/sub-*/subco_volume.txt")
    files_subco_control = glob.glob(str(path_roi_control) + "/sub-*/subco_volume.txt")

    # CLBP
    df_roi_clbp = {os.path.basename(os.path.dirname(files_subco_clbp[i])) : pd.read_csv(files_subco_clbp[i], sep='\t') for i in range(len(files_subco_clbp))}
    df_roi_clbp = pd.concat(df_roi_clbp)
    df_roi_clbp = df_roi_clbp.reset_index().rename(columns={'level_0': 'participant_id'}).drop(["Measure:volume", "level_1"], axis=1)
    df_roi_clbp[['participant_id', 'session']] = df_roi_clbp['participant_id'].str.rsplit('_ses-', 1, expand=True)
    df_roi_clbp = df_roi_clbp.merge(df_participants, on="participant_id")
    df_roi_clbp["group"] = "clbp"


    # Control
    df_roi_control = {os.path.basename(os.path.dirname(files_subco_control[i])): pd.read_csv(files_subco_control[i], sep='\t')
                   for i in range(len(files_subco_control))}
    df_roi_control = pd.concat(df_roi_control)
    df_roi_control = df_roi_control.reset_index().rename(columns={'level_0': 'participant_id'}).drop(
        ["Measure:volume", "level_1"], axis=1)
    df_roi_control['participant_id'] = df_roi_control['participant_id'].str.rsplit('_T1w', 0).str[0]
    df_roi_control[['participant_id', 'session']] = df_roi_control['participant_id'].str.rsplit('_ses-', 1, expand=True)
    df_roi_control = df_roi_control.merge(df_participants, on="participant_id")
    df_roi_control["group"] = "control"

    df_sub = pd.concat([df_roi_clbp, df_roi_control])
    return df_sub


def zscores(df_thick_lh, df_thick_rh, v_ses='v1'):
    mean_lh_clbp = df_thick_lh.set_index(['group', 'session']).xs(['clbp', v_ses]).filter(regex='lh(.+)_L(.+)',
                                                                                         axis=1).mean().values
    mean_lh_control = df_thick_lh.set_index(['group', 'session']).xs(['control', v_ses]).filter(regex='lh(.+)_L(.+)',
                                                                                               axis=1).mean().values
    std_lh_control = df_thick_lh.set_index(['group', 'session']).xs(['control', v_ses]).filter(regex='lh(.+)_L(.+)',
                                                                                              axis=1).std().values
    z_score_lh = (mean_lh_clbp - mean_lh_control) / std_lh_control
    mean_rh_clbp = df_thick_rh.set_index(['group', 'session']).xs(['clbp', v_ses]).filter(regex='rh(.+)_R(.+)',
                                                                                         axis=1).mean().values
    mean_rh_control = df_thick_rh.set_index(['group', 'session']).xs(['control', v_ses]).filter(regex='rh(.+)_R(.+)',
                                                                                               axis=1).mean().values
    std_rh_control = df_thick_rh.set_index(['group', 'session']).xs(['control', v_ses]).filter(regex='rh(.+)_R(.+)',
                                                                                              axis=1).std().values
    z_score_rh = (mean_rh_clbp - mean_rh_control) / std_rh_control
    zscore_thikness = [val for pair in zip(z_score_lh, z_score_rh) for val in pair]
    plot_point_brain(zscore_thikness, load_brainnetome_centroids()[:210], views='ax', views_size=(8, 4.8),
                     cmap='RdYlBu', vmin=-2, vmax=2)
    plt.show()

def zscores_sub(df_thick_lh, df_thick_rh, df_vol, v_ses='v1'):
    mean_lh_clbp = df_thick_lh.set_index(['group', 'session']).xs(['clbp', v_ses]).filter(regex='lh(.+)_L(.+)',
                                                                                          axis=1).mean().values
    mean_lh_control = df_thick_lh.set_index(['group', 'session']).xs(['control', v_ses]).filter(regex='lh(.+)_L(.+)',
                                                                                                axis=1).mean().values
    std_lh_control = df_thick_lh.set_index(['group', 'session']).xs(['control', v_ses]).filter(regex='lh(.+)_L(.+)',
                                                                                               axis=1).std().values
    z_score_lh = (mean_lh_clbp - mean_lh_control) / std_lh_control
    mean_rh_clbp = df_thick_rh.set_index(['group', 'session']).xs(['clbp', v_ses]).filter(regex='rh(.+)_R(.+)',
                                                                                          axis=1).mean().values
    mean_rh_control = df_thick_rh.set_index(['group', 'session']).xs(['control', v_ses]).filter(regex='rh(.+)_R(.+)',
                                                                                                axis=1).mean().values
    std_rh_control = df_thick_rh.set_index(['group', 'session']).xs(['control', v_ses]).filter(regex='rh(.+)_R(.+)',
                                                                                               axis=1).std().values
    z_score_rh = (mean_rh_clbp - mean_rh_control) / std_rh_control
    zscore_thikness = [val for pair in zip(z_score_lh, z_score_rh) for val in pair]
    mean_clbp = df_vol.set_index(['group', 'session']).xs(['clbp', v_ses]).filter(regex='(.+)_(.+)',
                                                                                         axis=1).mean().values
    mean_control = df_vol.set_index(['group', 'session']).xs(['control', v_ses]).filter(regex='(.+)_(.+)',
                                                                                               axis=1).mean().values
    std_control = df_vol.set_index(['group', 'session']).xs(['control', v_ses]).filter(regex='(.+)_(.+)',
                                                                                              axis=1).std().values
    z_score = (mean_clbp - mean_control) / std_control
    plot_point_brain(np.append(zscore_thikness, z_score), load_brainnetome_centroids(), views='ax', views_size=(8, 4.8),
                     cmap='RdYlBu', vmin=-2, vmax=2)
    plt.show()
    print(np.append(zscore_thikness, z_score).shape)
    print(np.argmax(np.append(zscore_thikness, z_score)))


def load_brainnetome_centroids():
    bn_centroids = get_centroids("/home/pabaua/dev_tpil/data/BN/BN_Atlas_for_FSL/Brainnetome/BNA-maxprob-thr0-1mm.nii.gz")
    return bn_centroids


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
    df_participants = pd.read_csv("/home/pabaua/dev_tpil/data/22-08-19_dMRI_CLBP_BIDS/participants.tsv", sep='\t')
    df_atlas = pd.read_csv("/home/pabaua/dev_tpil/data/BN/BN_atlas_legend.csv", sep=',')
    df_atlas['Lobe'].ffill(inplace=True)
    df_atlas['Gyrus'].ffill(inplace=True)
    df_atlas[['ROI_name', 'Modified cyto-architectonic']] = df_atlas['Modified cyto-architectonic'].str.partition(',', expand=True)[[0,2]]
    df_atlas['ROI_name'] = df_atlas['ROI_name'].str.split('/').str[-1]
    df_atlas['ROI_name'] = df_atlas['ROI_name'].str.split('and ').str[-1]
    path_output = os.path.abspath(arguments.o)

    # analysis
    df_vol_lh, df_vol_rh = cortical_volume_analysis_2(path_roi_clbp, path_roi_control, df_participants)
    zscores(df_vol_lh, df_vol_rh, v_ses='v3')
    #df_vol = cortical_volume_analysis(path_roi_clbp, path_roi_control, df_participants)
    # df_vol.columns.name = 'ROI_name'
    # df_vol = df_vol.stack()
    # df_vol = df_vol.to_frame().join(df_atlas.set_index('ROI_name')).reset_index()
    # df_vol = df_vol.groupby(['group', 'participant_id', 'age', 'sex', 'session', 'hemisphere', 'Lobe', 'Gyrus', 'ROI_name', 'Modified cyto-architectonic', 'lh.MNI (X,Y,Z)', 'rh.MNI (X, Y, Z)', 'Label ID.L', 'Label ID.R']).mean()
    # df_vol = df_vol.rename(columns={0:'volume'}).reset_index().set_index(['group', 'participant_id', 'age', 'sex', 'session', 'Lobe', 'Gyrus', 'ROI_name', 'Modified cyto-architectonic', 'lh.MNI (X,Y,Z)', 'rh.MNI (X, Y, Z)', 'Label ID.L', 'Label ID.R']).pivot(columns='hemisphere')
    # df_vol.to_csv("/home/pabaua/dev_tpil/data/Freesurfer/dataframe/BN_cortical_volume.csv")

    df_thick_lh, df_thick_rh = cortical_thick_analysis(path_roi_clbp, path_roi_control, df_participants)
    zscores(df_thick_lh, df_thick_rh, v_ses='v3')
    df_thick_lh.to_csv("/home/pabaua/dev_tpil/data/Freesurfer/dataframe/BN_cortical_thickness_lh.csv")
    df_thick_rh.to_csv("/home/pabaua/dev_tpil/data/Freesurfer/dataframe/BN_cortical_thickness_rh.csv")

    df_vol_sub = subcortical_analysis(path_roi_clbp, path_roi_control, df_participants)
    zscores_sub(df_vol_lh, df_vol_rh,df_vol_sub, v_ses='v3')
    df_vol_sub.to_csv("/home/pabaua/dev_tpil/data/Freesurfer/dataframe/BN_subcortical_volume.csv")


if __name__ == "__main__":
    main()