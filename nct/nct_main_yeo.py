
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
import matplotlib.pyplot as plt
import os
from pprint import pprint
import glob
from os.path import dirname as up


from functions.tpil_stats_freesurfer import cortical_volumes


#TODO: add cognitive topographies from neurosynth or brainmap
#TODO: test different meta-analysis techniques
#TODO: test different time horizons
#TODO: look up the activation values (could be binary)


def plot_points(values):
    print(values['v1'])
    brainnetome_atlas = nib.load("/home/pabaua/dev_tpil/results/results_connectivity_prep/test_container/results/sub-pl007_ses-v1/labels_in_mni.nii.gz")
    coords = plotting.find_parcellation_cut_coords(brainnetome_atlas)[:210]
    plotting.plot_markers(values['v1'], coords, title="Volume z-score per node (CLBP - control) v1", node_vmin=-2, node_vmax=2, node_cmap='RdYlBu', alpha=0.9)
    plt.show()
    plotting.plot_markers(values['v2'], coords, title="Volume z-score per node (CLBP - control) v2", node_vmin=-2,node_vmax=2, node_cmap='RdYlBu', alpha=0.9)
    plt.show()
    plotting.plot_markers(values['v3'], coords, title="Volume z-score per node (CLBP - control) v3", node_vmin=-2,node_vmax=2, node_cmap='RdYlBu', alpha=0.9)
    plt.show()


def plot_heatmap(x_con, x_clbp):
    # compute energy transition for each subject
    x_con = x_con[1].apply(lambda x: integrate_u(x)).apply(lambda x: np.sum(x))
    x_clbp = x_clbp[1].apply(lambda x: integrate_u(x)).apply(lambda x: np.sum(x))
    # figure
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(16, 3))
    df_g1 = (x_con.groupby(['x0', 'xf']).mean() / max(x_con.groupby(['x0', 'xf']).mean().tolist()))
    g1 = sns.heatmap(df_g1.unstack(level=-1),annot=True, ax=ax[0], cmap='Blues', square=True, vmin=0, vmax=1)
    g1.set_yticklabels(g1.get_yticklabels(), rotation=0, fontsize=8)
    g1.set_xticklabels(g1.get_xticklabels(), rotation=45, fontsize=8)
    ax[0].set(ylabel='', xlabel='', title='transition energy for controls')
    df_g2 = (x_clbp.groupby(['x0', 'xf']).mean() / max(x_clbp.groupby(['x0', 'xf']).mean().tolist()))
    g2 = sns.heatmap(df_g2.unstack(level=-1), annot=True, ax=ax[1], cmap='Blues', square=True, vmin=0, vmax=1)
    g2.set_yticklabels(g2.get_yticklabels(), rotation=0, fontsize=8)
    g2.set_xticklabels(g2.get_xticklabels(), rotation=45, fontsize=8)
    ax[1].set(ylabel='', xlabel='', title='transition energy for CLBPs')
    df_zscore = (x_clbp.groupby(['x0', 'xf']).mean() - x_con.groupby(['x0', 'xf']).mean()) / x_con.groupby(['x0', 'xf']).std()
    g3 = sns.heatmap(df_zscore.unstack(level=-1), annot=True, ax=ax[2], vmin=-2, vmax=2, cmap='RdBu', square=True)
    g3.set_yticklabels(g3.get_yticklabels(), rotation=0, fontsize=8)
    g3.set_xticklabels(g3.get_xticklabels(), rotation=45, fontsize=8)
    ax[2].set(ylabel='', xlabel='', title='transition energy z-score (CLBP - control)')
    plt.show()


def main():
    """
    main function, gather stats and call plots
    """
    path_fs_roi_clbp = '/home/pabaua/dev_tpil/data/Freesurfer/22-09-21_t1_clbp_freesurfer_output'
    path_fs_roi_control = '/home/pabaua/dev_tpil/data/Freesurfer/23_02_09_control_freesurfer_output'

    diff, z_score = cortical_volumes(path_fs_roi_clbp)
    #plot_points(z_score)


    # cres_atlas = apply_atlas(cres_dict)
    cres_atlas = get_parcellation()

    # get connectivity and apply network control theory
    A = load_connectivity()
    A_con_norm, A_clbp_norm = normalize_connectomes(A)
    #x_con_v1, x_clbp_v1 = get_state_traj(A_con_norm['v1'], A_clbp_norm['v1'], cres_atlas, out_dir, vol_corr=diff['v1'], ses='v1_')
    x_con_v2, x_clbp_v2 = get_state_traj(A_con_norm['v2'], A_clbp_norm['v2'], cres_atlas, out_dir,vol_corr=diff['v2'], ses='v2_')
    #x_con_v3, x_clbp_v3 = get_state_traj(A_con_norm['v3'], A_clbp_norm['v3'], cres_atlas, out_dir,vol_corr=diff['v3'], ses='v3_')
    plot_heatmap(x_con_v2, x_clbp_v2)

if __name__ == "__main__":
    main()