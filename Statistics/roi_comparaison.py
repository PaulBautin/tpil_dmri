
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
from netneurotools.datasets import fetch_conte69
from toolz import interleave

from brainspace.utils.parcellation import map_to_labels
from brainspace.plotting import plot_hemispheres, surface_plotting
import nibabel as nib
from brainspace import mesh
from netneurotools.utils import get_centroids


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


def subcortical_analysis(path_roi_clbp, path_roi_control, df_participants):
    files_subco_clbp = glob.glob(str(path_roi_clbp) + "/sub-*/subco_volume.txt")
    files_subco_control = glob.glob(str(path_roi_control) + "/sub-*/subco_volume.txt")

    # CLBP
    df_roi_clbp = {os.path.basename(os.path.dirname(files_subco_clbp[i])) : pd.read_csv(files_subco_clbp[i], sep='\t') for i in range(len(files_subco_clbp))}
    df_roi_clbp = pd.concat(df_roi_clbp)
    df_roi_clbp = df_roi_clbp.reset_index().rename(columns={'level_0': 'participant_id'}).drop(["Measure:volume", "level_1"], axis=1)
    df_roi_clbp[['participant_id', 'session']] = df_roi_clbp['participant_id'].str.rsplit('_ses-', 1, expand=True)
    print(df_roi_clbp.groupby(['session']).mean()['NAC_L'])
    print(df_roi_clbp.groupby(['session']).std()['NAC_L'])

    # Control
    df_roi_control = {os.path.basename(os.path.dirname(files_subco_control[i])): pd.read_csv(files_subco_control[i], sep='\t')
                   for i in range(len(files_subco_control))}
    df_roi_control = pd.concat(df_roi_control)
    df_roi_control = df_roi_control.reset_index().rename(columns={'level_0': 'participant_id'}).drop(
        ["Measure:volume", "level_1"], axis=1)
    df_roi_control['participant_id'] = df_roi_control['participant_id'].str.rsplit('_T1w', 0).str[0]
    df_roi_control[['participant_id', 'session']] = df_roi_control['participant_id'].str.rsplit('_ses-', 1, expand=True)
    print(df_roi_control.groupby(['session']).mean()['NAC_L'])
    print(df_roi_control.groupby(['session']).std()['NAC_L'])

    # Z_score
    df_zscore = (df_roi_clbp.groupby(['session']).mean() - df_roi_control.groupby(['session']).mean()) / df_roi_control.groupby(['session']).std()
    z_score_v1 = df_zscore.loc['v1'].values
    z_score_v2 = df_zscore.loc['v2'].values
    z_score_v3 = df_zscore.loc['v3'].values

    # Brainnetome plot
    brainnetome = nib.load("/home/pabaua/dev_tpil/data/BN/BN_Atlas_for_FSL/Brainnetome/BNA-maxprob-thr0-1mm.nii.gz")
    centroids = get_centroids(brainnetome)
    fig = plt.figure(figsize=(11, 5))

    ax = fig.add_subplot(131, projection='3d')
    ax.scatter(centroids[:210, 0], centroids[:210, 1], centroids[:210, 2], c=np.ones(210), s=15, alpha=0.1, cmap="gray")
    ax.scatter(centroids[210:, 0], centroids[210:, 1], centroids[210:, 2], s=30, c=z_score_v1, alpha=1, cmap='viridis', vmin=-1.5, vmax=1.5)
    ax.view_init(elev=0, azim=180)
    ax.axis('off')
    ax.set_box_aspect((np.ptp(centroids[:, 0]), np.ptp(centroids[:, 1]), np.ptp(centroids[:, 2])))

    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(centroids[:210, 0], centroids[:210, 1], centroids[:210, 2], c=np.ones(210), s=15, alpha=0.1,
                cmap="gray")
    ax2.scatter(centroids[210:, 0], centroids[210:, 1], centroids[210:, 2], s=30, c=z_score_v1, alpha=1,
                           cmap='viridis', vmin=-1.5, vmax=1.5)
    ax2.view_init(elev=90, azim=-90)
    ax2.axis('off')
    ax2.set_box_aspect((np.ptp(centroids[:, 0]), np.ptp(centroids[:, 1]), np.ptp(centroids[:, 2])))

    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(centroids[:210, 0], centroids[:210, 1], centroids[:210, 2], c=np.ones(210), s=15, alpha=0.1, cmap="gray")
    scatter3 = ax3.scatter(centroids[210:, 0], centroids[210:, 1], centroids[210:, 2], s=30, c=z_score_v1, alpha=1, cmap='viridis', vmin=-1.5, vmax=1.5)
    ax3.view_init(elev=0, azim=0)
    ax3.axis('off')
    ax3.set_box_aspect((np.ptp(centroids[:, 0]), np.ptp(centroids[:, 1]), np.ptp(centroids[:, 2])))

    #fig.colorbar(scatter3)
    fig.tight_layout()
    plt.show()


    # df_roi_data = pd.concat([df_roi_control, df_roi_clbp])
    # sns.violinplot(y=df_roi_data['Left-Inf-Lat-Vent'], x=df_roi_data['session'], hue=df_roi_data['group'])
    # Find center of mass of each image





def cortical_volume_analysis(path_roi_clbp, path_roi_control, df_participants):
    files_vol_clbp = glob.glob(str(path_roi_clbp) + "/sub-*/lh.volume.txt")
    files_vol_clbp_rh = glob.glob(str(path_roi_clbp) + "/sub-*/rh.volume.txt")
    files_vol_control = glob.glob(str(path_roi_control) + "/sub-*/lh.volume.txt")
    files_vol_control_rh = glob.glob(str(path_roi_control) + "/sub-*/rh.volume.txt")
    df_participants['participant_id'] = df_participants['participant_id'].str.replace('-', '-pl')

    # Get all data files and read json files into a dataframe
    df_roi_data_clbp = pd.concat(pd.read_csv(files_vol_clbp[i], sep='\t') for i in range(len(files_vol_clbp)))
    df_roi_data_clbp[['participant_id', 'session']] = df_roi_data_clbp['lh.BN_Atlas.volume'].str.rsplit('_ses-', 1,
                                                                                                        expand=True)
    df_roi_data_clbp_rh = pd.concat(pd.read_csv(files_vol_clbp_rh[i], sep='\t') for i in range(len(files_vol_clbp_rh)))
    df_vol_clbp = pd.concat([df_roi_data_clbp, df_roi_data_clbp_rh], axis=1)[list(interleave([df_roi_data_clbp, df_roi_data_clbp_rh]))]
    #df_vol_clbp = df_vol_clbp.merge(df_participants, on="participant_id")
    df_vol_clbp["group"] = "clbp"
    df_vol_clbp = df_vol_clbp.drop(["lh.BN_Atlas.volume", "rh.BN_Atlas.volume"], axis=1)
    print(df_vol_clbp.groupby(['session']).mean()["lh_A10m_L_volume"])
    print(df_vol_clbp.groupby(['session']).std()["lh_A10m_L_volume"])

    df_roi_data_control = pd.concat(pd.read_csv(files_vol_control[i], sep='\t') for i in range(len(files_vol_control)))
    df_roi_data_control['lh.BN_Atlas.volume'] = df_roi_data_control['lh.BN_Atlas.volume'].str.rsplit('_T1w', 0).str[0]
    df_roi_data_control[['participant_id', 'session']] = df_roi_data_control['lh.BN_Atlas.volume'].str.rsplit('_ses-',
                                                                                                              1,
                                                                                                              expand=True)
    df_roi_data_control_rh = pd.concat(pd.read_csv(files_vol_control_rh[i], sep='\t') for i in range(len(files_vol_control_rh)))
    df_roi_data_control_rh['rh.BN_Atlas.volume'] = df_roi_data_control_rh['rh.BN_Atlas.volume'].str.rsplit('_T1w', 0).str[0]
    df_vol_control = pd.concat([df_roi_data_control, df_roi_data_control_rh], axis=1)[
        list(interleave([df_roi_data_control, df_roi_data_control_rh]))]
    #df_vol_control = df_vol_control.merge(df_participants, on="participant_id")
    df_vol_control["group"] = "con"
    df_vol_control = df_vol_control.drop(["lh.BN_Atlas.volume", "rh.BN_Atlas.volume"], axis=1)
    print(df_vol_control.groupby(['session']).mean()["lh_A10m_L_volume"])
    print(df_vol_control.groupby(['session']).std()["lh_A10m_L_volume"])


    df_zscore = (df_vol_clbp.groupby(['session']).mean() - df_vol_control.groupby(['session']).mean()) / df_vol_control.groupby(['session']).std()
    df_roi_data = pd.concat([df_vol_control, df_vol_clbp])
    # sns.violinplot(y=df_roi_data['lh_A22r_L_volume'], x=df_roi_data['session'], hue=df_roi_data['group'])
    # plt.show()

    lhsurface, rhsurface = fetch_conte69()['midthickness'].lh, fetch_conte69()['midthickness'].rh
    lhlabels = "/home/pabaua/dev_tpil/data/BN/BN_Atlas_freesurfer/fsaverage/fsaverage_LR32k/fsaverage.L.BN_Atlas.32k_fs_LR.label.gii"
    rhlabels = "/home/pabaua/dev_tpil/data/BN/BN_Atlas_freesurfer/fsaverage/fsaverage_LR32k/fsaverage.R.BN_Atlas.32k_fs_LR.label.gii"
    atlas_labels = np.concatenate((nib.load(lhlabels).agg_data(), nib.load(rhlabels).agg_data())).astype("float")

    surf_z_score_v1 = map_to_labels(df_zscore.loc['v1'][:210].values, atlas_labels, mask=atlas_labels > 0, fill=np.nan)
    surf_z_score_v2 = map_to_labels(df_zscore.loc['v2'][:210].values, atlas_labels, mask=atlas_labels > 0, fill=np.nan)
    surf_z_score_v3 = map_to_labels(df_zscore.loc['v3'][:210].values, atlas_labels, mask=atlas_labels > 0, fill=np.nan)
    lhsurf = mesh.mesh_io.read_surface(lhsurface)
    rhsurf = mesh.mesh_io.read_surface(rhsurface)
    plot_hemispheres(lhsurf, rhsurf, [surf_z_score_v1, surf_z_score_v2, surf_z_score_v3], size=(1200, 600),cmap='viridis', color_bar=True,
                     label_text=["0 months", "2 months", "4 months"], color_range=(-1.5, 1.5))
    return df_zscore


def cortical_thickness_analysis(path_roi_clbp, path_roi_control, df_participants):
    files_vol_clbp = glob.glob(str(path_roi_clbp) + "/sub-*/lh.thickness.txt")
    files_vol_clbp_rh = glob.glob(str(path_roi_clbp) + "/sub-*/rh.thickness.txt")
    files_vol_control = glob.glob(str(path_roi_control) + "/sub-*/lh.thickness.txt")
    files_vol_control_rh = glob.glob(str(path_roi_control) + "/sub-*/rh.thickness.txt")
    df_participants['participant_id'] = df_participants['participant_id'].str.replace('-', '-pl')

    # Get all data files and read json files into a dataframe
    df_roi_data_clbp = pd.concat(pd.read_csv(files_vol_clbp[i], sep='\t') for i in range(len(files_vol_clbp)))
    df_roi_data_clbp[['participant_id', 'session']] = df_roi_data_clbp['lh.BN_Atlas.thickness'].str.rsplit('_ses-', 1,
                                                                                                        expand=True)
    df_roi_data_clbp_rh = pd.concat(pd.read_csv(files_vol_clbp_rh[i], sep='\t') for i in range(len(files_vol_clbp_rh)))
    df_vol_clbp = pd.concat([df_roi_data_clbp, df_roi_data_clbp_rh], axis=1)[list(interleave([df_roi_data_clbp, df_roi_data_clbp_rh]))]
    #df_vol_clbp = df_vol_clbp.merge(df_participants, on="participant_id")
    df_vol_clbp["group"] = "clbp"
    df_vol_clbp = df_vol_clbp.drop(["lh.BN_Atlas.thickness", "rh.BN_Atlas.thickness"], axis=1)
    print(df_vol_clbp.groupby(['session']).mean()["lh_A10m_L_thickness"])
    print(df_vol_clbp.groupby(['session']).std()["lh_A10m_L_thickness"])

    df_roi_data_control = pd.concat(pd.read_csv(files_vol_control[i], sep='\t') for i in range(len(files_vol_control)))
    df_roi_data_control['lh.BN_Atlas.thickness'] = df_roi_data_control['lh.BN_Atlas.thickness'].str.rsplit('_T1w', 0).str[0]
    df_roi_data_control[['participant_id', 'session']] = df_roi_data_control['lh.BN_Atlas.thickness'].str.rsplit('_ses-',
                                                                                                              1,
                                                                                                              expand=True)
    df_roi_data_control_rh = pd.concat(pd.read_csv(files_vol_control_rh[i], sep='\t') for i in range(len(files_vol_control_rh)))
    df_roi_data_control_rh['rh.BN_Atlas.thickness'] = df_roi_data_control_rh['rh.BN_Atlas.thickness'].str.rsplit('_T1w', 0).str[0]
    df_vol_control = pd.concat([df_roi_data_control, df_roi_data_control_rh], axis=1)[
        list(interleave([df_roi_data_control, df_roi_data_control_rh]))]
    #df_vol_control = df_vol_control.merge(df_participants, on="participant_id")
    df_vol_control["group"] = "con"
    print(df_vol_control.groupby(['session']).mean()["lh_A10m_L_thickness"])
    print(df_vol_control.groupby(['session']).std()["lh_A10m_L_thickness"])
    df_vol_control = df_vol_control.drop(["lh.BN_Atlas.thickness", "rh.BN_Atlas.thickness"], axis=1)
    df_zscore = (df_vol_clbp.groupby(['session']).mean() - df_vol_control.groupby(['session']).mean()) / df_vol_control.groupby(['session']).std()
    df_roi_data = pd.concat([df_vol_control, df_vol_clbp])
    # sns.violinplot(y=df_roi_data['lh_A22r_L_volume'], x=df_roi_data['session'], hue=df_roi_data['group'])
    # plt.show()

    lhsurface, rhsurface = fetch_conte69()['midthickness'].lh, fetch_conte69()['midthickness'].rh
    lhlabels = "/home/pabaua/dev_tpil/data/BN/BN_Atlas_freesurfer/fsaverage/fsaverage_LR32k/fsaverage.L.BN_Atlas.32k_fs_LR.label.gii"
    rhlabels = "/home/pabaua/dev_tpil/data/BN/BN_Atlas_freesurfer/fsaverage/fsaverage_LR32k/fsaverage.R.BN_Atlas.32k_fs_LR.label.gii"
    atlas_labels = np.concatenate((nib.load(lhlabels).agg_data(), nib.load(rhlabels).agg_data())).astype("float")
    print(df_zscore.loc['v1'][:210].values)
    print(df_zscore.loc['v2'][:210].values)
    surf_z_score_v1 = map_to_labels(df_zscore.loc['v1'][:210].values, atlas_labels, mask=atlas_labels > 0, fill=np.nan)
    surf_z_score_v2 = map_to_labels(df_zscore.loc['v2'][:210].values, atlas_labels, mask=atlas_labels > 0, fill=np.nan)
    surf_z_score_v3 = map_to_labels(df_zscore.loc['v3'][:210].values, atlas_labels, mask=atlas_labels > 0, fill=np.nan)
    lhsurf = mesh.mesh_io.read_surface(lhsurface)
    rhsurf = mesh.mesh_io.read_surface(rhsurface)
    plot_hemispheres(lhsurf, rhsurf, [surf_z_score_v1, surf_z_score_v2, surf_z_score_v3], size=(1200, 600),cmap='viridis', color_bar=True,
                     label_text=["0 months", "2 months", "4 months"], color_range=(-1.5, 1.5))
    plt.show()
    return df_zscore


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
    path_output = os.path.abspath(arguments.o)

    # analysis
    subcortical_analysis(path_roi_clbp, path_roi_control, df_participants)
    cortical_volume_analysis(path_roi_clbp, path_roi_control, df_participants)
    cortical_thickness_analysis(path_roi_clbp, path_roi_control, df_participants)



if __name__ == "__main__":
    main()