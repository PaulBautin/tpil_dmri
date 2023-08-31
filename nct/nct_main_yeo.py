
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
import nibabel as nib
from nilearn import plotting, datasets
import pandas as pd
from nctpy.utils import matrix_normalization
from nilearn import maskers


from functions.tpil_stats_freesurfer import cortical_measures, cortical_measures_diff, cortical_measures_z_score
from functions.tpil_sc_load import load_connectivity, find_files_with_common_name


#TODO: add cognitive topographies from neurosynth or brainmap
#TODO: test different meta-analysis techniques
#TODO: test different time horizons
#TODO: look up the activation values (could be binary)


def plot_points(values, atlas, lut, metric='z_score'):
    df_z_score = pd.concat([values['lh'].stack(), values['rh'].stack()]).to_frame().reset_index().rename(columns={'level_1':'regions',0:metric}).set_index('regions')
    coords, labels_list = plotting.find_parcellation_cut_coords(atlas, return_label_names=True)
    print(labels_list)
    df_coords = pd.DataFrame({'coords':coords.tolist(), 'labels_list':labels_list}).set_index('labels_list')
    df_coords = df_coords.join(lut.set_index('label')).reset_index()
    df_coords['region_fs'] = df_coords['region'].str.split('/').str[-1]
    print(df_coords.set_index('region_fs'))
    df = df_coords.set_index('region_fs').join(df_z_score).dropna(axis=0)
    print(df)
    # plotting.plot_markers(values['v1'], coords, title="Volume z-score per node (CLBP - control) v1", node_vmin=-2, node_vmax=2, node_cmap='RdYlBu', alpha=0.9)
    # plt.show()
    # plotting.plot_markers(values['v2'], coords, title="Volume z-score per node (CLBP - control) v2", node_vmin=-2,node_vmax=2, node_cmap='RdYlBu', alpha=0.9)
    # plt.show()
    # plotting.plot_markers(values['v3'], coords, title="Volume z-score per node (CLBP - control) v3", node_vmin=-2,node_vmax=2, node_cmap='RdYlBu', alpha=0.9)
    # plt.show()


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


def normalize_connectomes(df_A):
    system = 'continuous'  # option 'discrete'
    df_A = df_A.drop('roi', axis=1)
    df_A_norm = df_A.groupby(['session', 'subject']).apply(lambda x: matrix_normalization(A=x.drop(['session','subject'], axis=1), c=1, system=system))
    return df_A_norm


def get_yeo_parcellation(atlas_img):
    def label_extractor(img_yeo, data_yeo, i):
        data_yeo_copy = data_yeo.copy()
        data_yeo_copy[data_yeo_copy != i] = 0
        data_yeo_copy[data_yeo_copy == i] = 1
        img_yeo_1 = nib.Nifti1Image(data_yeo_copy, img_yeo.affine, img_yeo.header)
        return img_yeo_1

   
    # strategy is the name of a valid function to reduce the region with
    roi = maskers.NiftiLabelsMasker(atlas_img, strategy='mean')
    networks = {1:'visual', 2:'somatomotor', 3:'dorsal attention', 4:'ventral attention', 5:'limbic', 6:'frontoparietal', 7:'default'}
    yeo = datasets.fetch_atlas_yeo_2011()
    img_yeo = nib.load(yeo.thick_7)
    data_yeo = img_yeo.get_fdata()
    img_dict = {i: label_extractor(img_yeo, data_yeo, i) for i in np.delete(np.unique(data_yeo), 0)}
    dict_signal = {networks[k]: roi.fit_transform(v)[:,:210] for k, v in img_dict.items()}
    #img_signal = {k: roi.inverse_transform(v) for k, v in dict_signal.items()}
    return dict_signal


def main():
    """
    main function, gather stats and call plots
    """
    path_fs_roi_clbp = '/home/pabaua/dev_tpil/data/Freesurfer/22-09-21_t1_clbp_freesurfer_output'
    path_fs_roi_con = '/home/pabaua/dev_tpil/data/Freesurfer/23_02_09_control_freesurfer_output'
    df_clbp = cortical_measures(path_fs_roi_clbp, atlas='Schaefer2018_200Parcels_7Networks.volume')
    df_con = cortical_measures(path_fs_roi_con, atlas='Schaefer2018_200Parcels_7Networks.volume')

    df_diff = {k: cortical_measures_diff(df_clbp[k], df_con[k]) for k in ['lh', 'rh']}
    df_z_score = {k: cortical_measures_z_score(df_clbp[k], df_con[k]) for k in ['lh', 'rh']}
    
    brainnetome_atlas = nib.load("/home/pabaua/dev_tpil/data/BN/BN_Atlas_for_FSL/Brainnetome/BNA-maxprob-thr0-1mm.nii.gz")
    brainnetome_lut = pd.read_csv("/home/pabaua/dev_tpil/data/BN/BN_Atlas_for_FSL/Brainnetome.lut", sep=" ", header=None).rename(columns={0: "label", 4: "region"}).drop([1,2,3], axis=1)
    #plot_points(df_z_score, brainnetome_atlas, brainnetome_lut)

    atlas_schaefer = datasets.fetch_atlas_schaefer_2018(n_rois=200, yeo_networks=7, resolution_mm=1)
    schaefer_atlas = nib.load(atlas_schaefer.maps)
    schaefer_lut = pd.DataFrame({'label':np.array(range(200))+1, 'region':atlas_schaefer.labels})
    schaefer_lut['region'] = schaefer_lut['region'].apply(lambda x: x.decode("utf-8"))
    print(schaefer_lut)
    #plot_points(df_z_score, schaefer_atlas, schaefer_lut)


    directory_schaefer = '/home/pabaua/dev_tpil/results/results_connectflow/23-08-17_connectflow/clbp_schaefer'
    common_name = 'commit2_weights.csv'
    labels_list = pd.read_csv('/home/pabaua/dev_tpil/tpil_connectivity_prep/freesurfer_data/atlas_schaefer_200_first_label_list.txt', header=None, dtype=np.int32)[0].to_numpy()
    df_A = find_files_with_common_name(directory_schaefer, common_name, labels_list)
    df_A_norm = normalize_connectomes(df_A)
    

    dict_signal = get_yeo_parcellation(schaefer_atlas)
    print(dict_signal)


    #x_con_v1, x_clbp_v1 = get_state_traj(A_con_norm['v1'], A_clbp_norm['v1'], cres_atlas, out_dir, vol_corr=diff['v1'], ses='v1_')
    x_con_v2, x_clbp_v2 = get_state_traj(A_con_norm['v2'], A_clbp_norm['v2'], cres_atlas, out_dir,vol_corr=diff['v2'], ses='v2_')
    #x_con_v3, x_clbp_v3 = get_state_traj(A_con_norm['v3'], A_clbp_norm['v3'], cres_atlas, out_dir,vol_corr=diff['v3'], ses='v3_')
    plot_heatmap(x_con_v2, x_clbp_v2)

if __name__ == "__main__":
    main()