
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
from nctpy.energies import get_control_inputs, integrate_u
from nctpy.metrics import ave_control
from nctpy.utils import (
    matrix_normalization,
    convert_states_str2int,
    normalize_state,
    normalize_weights,
    get_null_p,
    get_fdr_p,
)

from nilearn import maskers, plotting, image
import seaborn as sns
import matplotlib.cm as cm
import matplotlib as mpl
import plotly.express as px


from functions.tpil_stats_freesurfer import cortical_measures, cortical_measures_diff, cortical_measures_z_score
from functions.tpil_sc_load import load_connectivity, find_files_with_common_name
from functions.tpil_meta_analysis import fetch_neurosynth_data, get_studies_by_terms, apply_meta_analysis, apply_atlas_meta


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


def plot_heatmap(df_con, df_clbp, ses='v1'):
    df_con = df_con.groupby('session').get_group(ses).drop(['session','subject'], axis=1)
    df_clbp = df_clbp.groupby('session').get_group(ses).drop(['session','subject'], axis=1)
    # figure
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(16, 3), sharex=True, sharey=True)
    print(df_con.groupby('x0').mean())
    print(np.max(df_con.groupby('x0').mean().values))
    df_g1 = (df_con.groupby('x0').mean() / np.max(df_con.groupby('x0').mean().values))
    print(df_g1)
    g1 = sns.heatmap(df_g1,annot=True, ax=ax[0], cmap='Blues', square=True, vmin=0, vmax=1)
    g1.set_yticklabels(g1.get_yticklabels(), rotation=0, fontsize=12)
    g1.set_xticklabels(g1.get_xticklabels(), rotation=45, fontsize=12, ha='right')
    ax[0].set(ylabel='', xlabel='', title='transition energy for controls')
    df_g2 = (df_clbp.groupby('x0').mean() / np.max(df_clbp.groupby('x0').mean().values))
    g2 = sns.heatmap(df_g2, annot=True, ax=ax[1], cmap='Blues', square=True, vmin=0, vmax=1)
    #g2.set_yticklabels(g2.get_yticklabels(), rotation=0, fontsize=12)
    g2.set_xticklabels(g2.get_xticklabels(), rotation=45, fontsize=12, ha='right')
    ax[1].set(ylabel='', xlabel='', title='transition energy for CLBPs')
    df_zscore = (df_clbp.groupby('x0').mean() - df_con.groupby('x0').mean()) / df_con.groupby('x0').std()
    g3 = sns.heatmap(df_zscore, annot=True, ax=ax[2], vmin=-2, vmax=2, cmap='RdBu', square=True)
    #g3.set_yticklabels(g3.get_yticklabels(), rotation=0, fontsize=12)
    g3.set_xticklabels(g3.get_xticklabels(), rotation=45, fontsize=12, ha='right')
    ax[2].set(ylabel='', xlabel='', title='transition energy z-score (CLBP - control)')
    plt.show()
    

def plot_heatmap_v1v2v3(df_con_all, df_clbp_all):
    df_con = df_con_all.groupby('session').get_group('v1').drop(['session','subject'], axis=1)
    df_clbp = df_clbp_all.groupby('session').get_group('v1').drop(['session','subject'], axis=1)
    # figure
    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(28, 21), sharex=True, sharey=True)
    df_g1 = (df_con.groupby('x0').mean() / np.max(df_con.groupby('x0').mean().values))
    print(df_g1)
    g11 = sns.heatmap(df_g1,annot=True, ax=ax[0,0], cmap='Blues', vmin=0, vmax=1)
    g11.set_yticklabels(g11.get_yticklabels(), rotation=0, fontsize=12)
    g11.set_xticklabels(g11.get_xticklabels(), rotation=45, fontsize=12)
    ax[0,0].set(ylabel='', xlabel='', title='transition energy for controls v1')
    df_g2 = (df_clbp.groupby('x0').mean() / np.max(df_clbp.groupby('x0').mean().values))
    g12 = sns.heatmap(df_g2, annot=True, ax=ax[0,1], cmap='Blues', vmin=0, vmax=1)
    g12.set_yticklabels(g12.get_yticklabels(), rotation=0, fontsize=12)
    g12.set_xticklabels(g12.get_xticklabels(), rotation=45, fontsize=12)
    ax[0,1].set(ylabel='', xlabel='', title='transition energy for CLBPs v1')
    df_zscore = (df_clbp.groupby('x0').mean() - df_con.groupby('x0').mean()) / df_con.groupby('x0').std()
    g13 = sns.heatmap(df_zscore, annot=True, ax=ax[0,2], vmin=-2, vmax=2, cmap='RdBu')
    g13.set_yticklabels(g13.get_yticklabels(), rotation=0, fontsize=12)
    g13.set_xticklabels(g13.get_xticklabels(), rotation=45, fontsize=12)
    ax[0,2].set(ylabel='', xlabel='', title='transition energy z-score (CLBP - control) v1')

    # figure
    df_con = df_con_all.groupby('session').get_group('v2').drop(['session','subject'], axis=1)
    df_clbp = df_clbp_all.groupby('session').get_group('v2').drop(['session','subject'], axis=1)
    df_g1 = (df_con.groupby('x0').mean() / np.max(df_con.groupby('x0').mean().values))
    print(df_g1)
    g21 = sns.heatmap(df_g1,annot=True, ax=ax[1,0], cmap='Blues', vmin=0, vmax=1)
    g21.set_yticklabels(g21.get_yticklabels(), rotation=0, fontsize=12)
    g21.set_xticklabels(g21.get_xticklabels(), rotation=45, fontsize=12)
    ax[1,0].set(ylabel='', xlabel='', title='transition energy for controls v2')
    df_g2 = (df_clbp.groupby('x0').mean() / np.max(df_clbp.groupby('x0').mean().values))
    g22 = sns.heatmap(df_g2, annot=True, ax=ax[1,1], cmap='Blues', vmin=0, vmax=1)
    #g22.set_yticklabels(g22.get_yticklabels(), rotation=0, fontsize=12)
    g22.set_xticklabels(g22.get_xticklabels(), rotation=45, fontsize=12)
    ax[1,1].set(ylabel='', xlabel='', title='transition energy for CLBPs v2')
    df_zscore = (df_clbp.groupby('x0').mean() - df_con.groupby('x0').mean()) / df_con.groupby('x0').std()
    g23 = sns.heatmap(df_zscore, annot=True, ax=ax[1,2], vmin=-2, vmax=2, cmap='RdBu')
    #g23.set_yticklabels(g23.get_yticklabels(), rotation=0, fontsize=12)
    g23.set_xticklabels(g23.get_xticklabels(), rotation=45, fontsize=12)
    ax[1,2].set(ylabel='', xlabel='', title='transition energy z-score (CLBP - control) v2')

    # figure
    df_con = df_con_all.groupby('session').get_group('v3').drop(['session','subject'], axis=1)
    df_clbp = df_clbp_all.groupby('session').get_group('v3').drop(['session','subject'], axis=1)
    df_g1 = (df_con.groupby('x0').mean() / np.max(df_con.groupby('x0').mean().values))
    print(df_g1)
    g31 = sns.heatmap(df_g1,annot=True, ax=ax[2, 0], cmap='Blues', vmin=0, vmax=1)
    g31.set_yticklabels(g31.get_yticklabels(), rotation=0, fontsize=12)
    g31.set_xticklabels(g31.get_xticklabels(), rotation=30, fontsize=12, ha='right')
    ax[2, 0].set(ylabel='', xlabel='', title='transition energy for controls v3')
    df_g2 = (df_clbp.groupby('x0').mean() / np.max(df_clbp.groupby('x0').mean().values))
    g32 = sns.heatmap(df_g2, annot=True, ax=ax[2, 1], cmap='Blues', vmin=0, vmax=1)
    #g32.set_yticklabels(g32.get_yticklabels(), rotation=0, fontsize=12)
    g32.set_xticklabels(g32.get_xticklabels(), rotation=30, fontsize=12, ha='right')
    ax[2, 1].set(ylabel='', xlabel='', title='transition energy for CLBPs v3')
    df_zscore = (df_clbp.groupby('x0').mean() - df_con.groupby('x0').mean()) / df_con.groupby('x0').std()
    g33 = sns.heatmap(df_zscore, annot=True, ax=ax[2, 2], vmin=-2, vmax=2, cmap='RdBu')
    #g33.set_yticklabels(g33.get_yticklabels(), rotation=0, fontsize=12)
    g33.set_xticklabels(g33.get_xticklabels(), rotation=30, fontsize=12, ha='right')
    ax[2, 2].set(ylabel='', xlabel='', title='transition energy z-score (CLBP - control) v3')
    plt.show()


def plot_average_control(df_avg_con, df_avg_clbp, atlas, ses='v1'):
    coords, labels_list = plotting.find_parcellation_cut_coords(atlas, return_label_names=True)
    print(np.array(labels_list).shape)
    df_avg_con = df_avg_con[df_avg_con['session'] == ses]
    df_avg_clbp = df_avg_clbp[df_avg_clbp['session'] == ses]
    print(df_avg_clbp)
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(16, 15), sharex=True, sharey=True)
    plotting.plot_markers(df_avg_con.drop('subject', axis=1).groupby('session').mean(), coords, title="average control per node control", node_cmap='RdYlBu', alpha=0.9, axes=ax[0])
    plotting.plot_markers(df_avg_clbp.drop('subject', axis=1).groupby('session').mean(), coords, title="average control per node clbp", node_cmap='RdYlBu', alpha=0.9, axes=ax[1])
    df_zscore = (df_avg_clbp.drop('subject', axis=1).groupby('session').mean() - df_avg_con.drop('subject', axis=1).groupby('session').mean()) / df_avg_con.drop('subject', axis=1).groupby('session').std()
    plotting.plot_markers(df_zscore, coords, title="average control per node zscore", node_cmap='RdYlBu', alpha=0.9, axes=ax[2])
    plt.show()


def normalize_connectomes(df_A):
    system = 'continuous'  # option 'discrete'
    df_A = df_A.drop('roi', axis=1)
    #df_A_norm = df_A.groupby(['session', 'subject']).apply(lambda x: matrix_normalization(A=x.drop(['session','subject'], axis=1).iloc[-200:,-200:], c=1, system=system))
    df_A_norm = df_A.groupby(['session', 'subject']).apply(lambda x: matrix_normalization(A=x.drop(['session','subject'], axis=1), c=1, system=system))
    return df_A_norm.reset_index().drop(labels='level_2', axis=1)


def get_yeo_parcellation(atlas_img):
    def label_extractor(img_yeo, data_yeo, i):
        data_yeo_copy = data_yeo.copy()
        data_yeo_copy[data_yeo_copy != i] = 0
        data_yeo_copy[data_yeo_copy == i] = 1
        img_yeo_1 = nib.Nifti1Image(data_yeo_copy, img_yeo.affine, img_yeo.header)
        return img_yeo_1

   
    # strategy is the name of a valid function to reduce the region with
    roi = maskers.NiftiLabelsMasker(atlas_img, strategy='mean', background_label=0)
    networks = {1:'visual', 2:'somatomotor', 3:'dorsal attention', 4:'ventral attention', 5:'limbic', 6:'frontoparietal', 7:'default'}
    yeo = datasets.fetch_atlas_yeo_2011()
    img_yeo = nib.load(yeo.thick_7)
    data_yeo = img_yeo.get_fdata()
    img_dict = {i: label_extractor(img_yeo, data_yeo, i) for i in np.delete(np.unique(data_yeo), 0)}
    dict_signal = {networks[k]: roi.fit_transform(v) for k, v in img_dict.items()}
    img_signal = {k: roi.inverse_transform(v) for k, v in dict_signal.items()}
    plotting.plot_stat_map(img_signal['limbic'], title='limbic', display_mode='x', cut_coords=[-40, -20, 0, 20, 40], colorbar=True)
    plt.show()
    return dict_signal



def get_state_traj(df_A_norm, dict_states, out_dir, fname):
    # set parameters
    system = 'continuous'  # option 'discrete'
    T = 3  # time horizon
    rho = 1  # mixing parameter for state trajectory constraint
    S = np.eye(215)  # nodes in state trajectory to be constrained
    B = np.eye(215)  # which nodes receive input, uniform full control set
    if os.path.isfile(os.path.join(out_dir, fname)):
        df = pd.read_pickle(os.path.join(out_dir, fname))
    else:
        # df_A_norm = df_A_norm.groupby(['session']).get_group(('v1'))
        # df_A_norm = df_A_norm.groupby(['session','subject']).get_group(('v1','sub-pl002'))
        df = df_A_norm.groupby(['session', 'subject']).apply(lambda x: pd.Series({(x_f,x_i): np.sum(integrate_u(get_control_inputs(A_norm=x.drop(['session','subject', 'B'], axis=1), T=T, B=x['B'].mean(), x0=dict_states[x_i], xf=dict_states[x_f], system=system, rho=rho, S=S)[1])) for x_i in dict_states.keys() for x_f in dict_states.keys()}))
        df = df.stack().reset_index().rename(columns={'level_2':'x0'})
        df.to_pickle(os.path.join(out_dir, fname))
    return df


def get_states_from_meta_analysis(schaefer_atlas, out_dir):
    # get states from meta-analysis
    term_list = ["pain", "reward", "stress", "anxiety", "depression", "learning", "memory"]
    neurosynth_dset = fetch_neurosynth_data(out_dir)
    neurosynth_dset_by_term = get_studies_by_terms(neurosynth_dset, term_list)
    meta_dict = apply_meta_analysis(neurosynth_dset_by_term, out_dir)
    dict_states = apply_atlas_meta(meta_dict, schaefer_atlas)
    return dict_states


def get_states_from_yeo_atlas(out_dir):
    # load node-to-system mapping
    data_dir = '/home/pabaua/dev_tpil/data/Schaefer'
    system_labels = np.array(np.loadtxt(os.path.join(data_dir, "pnc_schaefer200_system_labels.txt"), dtype=str))
    # use list of system names to create states
    states, state_labels = convert_states_str2int(system_labels)
    state_dict = {k: np.append(np.zeros(15), normalize_state((system_labels == k).astype(int))) for k in state_labels}
    return state_dict
    

def apply_atlas(meta_dict, atlas):
    """
    Apply Nilearn NiftiLabelsMasker to the dictionnary of NiMARE results.MetaResults

    Parameters
    ----------
    meta_dict : dict
        Dictionary that contains terms of interest as keys and state map for each term as values
    atlas : nibabel.nifti1.Nifti1Image
        label atlas to use for the NiftiLabelsMasker

    Returns
    -------
    state_dict : dict
        Where keys are terms of interest and values are the corresponding normalized states numpy array (N_atlas_regions,)
    """
    # strategy is the name of a valid function to reduce the region with
    roi = maskers.NiftiLabelsMasker(atlas, strategy='mean')
    state_dict = {k: normalize_state(roi.fit_transform(v)).T[...,0] for k, v in meta_dict.items()}
    img_signal = {k: roi.inverse_transform(v) for k, v in state_dict.items()}
    plotting.plot_stat_map(img_signal['cPDM'], title='cPDM', display_mode='ortho', colorbar=True)
    plt.show()
    return state_dict


def compute_avg_control(df_A_norm ,labels_list, out_dir, fname):
    """
    Apply Nilearn NiftiLabelsMasker to the dictionnary of NiMARE results.MetaResults

    Parameters
    ----------
    meta_dict : dict
        Dictionary that contains terms of interest as keys and state map for each term as values
    atlas : nibabel.nifti1.Nifti1Image
        label atlas to use for the NiftiLabelsMasker

    Returns
    -------
    state_dict : dict
        Where keys are terms of interest and values are the corresponding normalized states numpy array (N_atlas_regions,)
    """
    if os.path.isfile(os.path.join(out_dir, fname)):
        df_avg_c = pd.read_pickle(os.path.join(out_dir, fname))
    else:
        df_avg_c = df_A_norm.groupby(['session', 'subject']).apply(lambda x: pd.Series(ave_control(A_norm=x.drop(['session','subject', 'B'], axis=1), system='continuous'), index=labels_list)).reset_index()
        df_avg_c.to_pickle(os.path.join(out_dir, fname))
    return df_avg_c





def main():
    """
    main function, gather stats and call plots
    """
    out_dir = os.path.abspath("/home/pabaua/example_data/example_data/")
    os.makedirs(out_dir, exist_ok=True)

    # get conversion between labels list and freesufer labels
    df_conversion = pd.read_json('/home/pabaua/dev_tpil/data/Schaefer/Schaefer2018_200Parcels_7Networks_order_LUT.json', orient='index', dtype=str)
    df_conversion.index = df_conversion.index.str.split('.').str[-1]
    dict_conversion = df_conversion.to_dict()[0]

    # get cortical measures
    path_fs_roi_clbp = '/home/pabaua/dev_tpil/data/Freesurfer/22-09-21_t1_clbp_freesurfer_output'
    path_fs_roi_con = '/home/pabaua/dev_tpil/data/Freesurfer/23_02_09_control_freesurfer_output'
    df_clbp = cortical_measures(path_fs_roi_clbp, atlas='Schaefer2018_200Parcels_7Networks.volume')
    df_clbp = pd.concat([df_clbp['lh'].set_index(['participant_id', 'session']), df_clbp['rh'].set_index(['participant_id', 'session'])], axis=1).reset_index()
    df_clbp = df_clbp.rename(dict_conversion, axis='columns')
    df_con = cortical_measures(path_fs_roi_con, atlas='Schaefer2018_200Parcels_7Networks.volume')
    df_con = pd.concat([df_con['lh'].set_index(['participant_id', 'session']), df_con['rh'].set_index(['participant_id', 'session'])], axis=1).reset_index()
    df_con = df_con.rename(dict_conversion, axis='columns')

    # get difference of cortical measures between groups
    df_diff_clbp, df_diff_con = cortical_measures_diff(df_clbp, df_con)
    df_B_clbp = df_diff_clbp.groupby(['session', 'participant_id']).apply(lambda x: np.append(np.zeros(15), x.drop(['session', 'participant_id'], axis=1).to_numpy()) * np.eye(215) + np.eye(215))
    df_B_con = df_diff_con.groupby(['session', 'participant_id']).apply(lambda x: np.append(np.zeros(15), x.drop(['session', 'participant_id'], axis=1).to_numpy()) * np.eye(215) + np.eye(215))
    #df_B_clbp = df_diff_clbp.groupby(['session', 'participant_id']).apply(lambda x: x.drop(['session', 'participant_id'], axis=1).to_numpy() * np.eye(200) + np.eye(200))
    #df_B_con = df_diff_con.groupby(['session', 'participant_id']).apply(lambda x: x.drop(['session', 'participant_id'], axis=1).to_numpy() * np.eye(200) + np.eye(200))

    #atlas_schaefer = datasets.fetch_atlas_schaefer_2018(n_rois=200, yeo_networks=7, resolution_mm=1)
    #schaefer_atlas = nib.load(atlas_schaefer.maps)
    # datasets.load_mni152_template(resolution=1)
    schaefer_atlas = nib.load('/home/pabaua/dev_tpil/sub-pl007_ses-v1__labels_warped_mni_int16.nii.gz')
    pain_dict = {'SIIPS1': nib.load('/home/pabaua/dev_tpil/data/Tor_wager_brain_signature/SIIPS1/nonnoc_v11_4_subcluster_maps_fdr05_pattern_wttest.nii.gz'),
                 'cPDM': nib.load('/home/pabaua/dev_tpil/data/Tor_wager_brain_signature/Geuter_2020/Geuter_2020_cPDM_combined_pain_map(1).nii.gz'),}
    #dict_states = apply_atlas(pain_dict, schaefer_atlas)

    #print(np.unique(schaefer_atlas.get_fdata()).shape)
    #dict_states = get_yeo_parcellation(schaefer_atlas)
    #dict_states.update(get_states_from_yeo_atlas(out_dir))
    #print(dict_states)
    dict_states = get_states_from_meta_analysis(schaefer_atlas, out_dir)
    #print(dict_states['anxiety'])



    # compute energy transitions for controls
    directory_schaefer = '/home/pabaua/dev_tpil/results/results_connectflow/23-08-21_connectflow/control_schaefer'
    common_name = 'commit2_weights.csv'
    labels_list = pd.read_csv('/home/pabaua/dev_tpil/tpil_connectivity_prep/freesurfer_data/atlas_schaefer_200_first_label_list.txt', header=None, dtype=np.int32)[0].to_numpy()
    df_A = find_files_with_common_name(directory_schaefer, common_name, labels_list)
    sns.heatmap(df_A.set_index(['session', 'subject']).loc['v1', 'sub-pl002'].values, vmin=0, vmax=10)
    plt.show()
    df_A_norm = normalize_connectomes(df_A).set_index(['session', 'subject'])
    df_A_norm['B'] = df_B_con
    df_A_norm = df_A_norm.reset_index()
    df_avg_c_con = compute_avg_control(df_A_norm, labels_list, out_dir, fname='df_avg_c_con.pkl')
    fname = 'df_control_schaefer_volume_sub_norm_tor.pkl'
    fname = 'df_control_schaefer_volume_sub_norm_neuro.pkl'
    df_energy_control_schaefer = get_state_traj(df_A_norm, dict_states, out_dir, fname)
    print(df_energy_control_schaefer)

    # compute energy transitions for clbps
    directory_schaefer = '/home/pabaua/dev_tpil/results/results_connectflow/23-08-21_connectflow/clbp_schaefer'
    common_name = 'commit2_weights.csv'
    labels_list = pd.read_csv('/home/pabaua/dev_tpil/tpil_connectivity_prep/freesurfer_data/atlas_schaefer_200_first_label_list.txt', header=None, dtype=np.int32)[0].to_numpy()
    df_A = find_files_with_common_name(directory_schaefer, common_name, labels_list)
    df_A_norm = normalize_connectomes(df_A).set_index(['session', 'subject'])
    df_A_norm['B'] = df_B_clbp
    df_A_norm = df_A_norm.reset_index()
    df_avg_c_clbp = compute_avg_control(df_A_norm, labels_list, out_dir, fname='df_avg_c_clbp.pkl')
    fname = 'df_clbp_schaefer_volume_sub_norm_tor.pkl'
    fname = 'df_clbp_schaefer_volume_sub_norm_neuro.pkl'
    df_energy_clbp_schaefer = get_state_traj(df_A_norm, dict_states, out_dir, fname)
    print(df_energy_clbp_schaefer)

    # plot heatmap of energy tansitions for controls and clbps and difference
    #plot_heatmap(df_energy_control_schaefer, df_energy_clbp_schaefer, ses='v1')
    plot_heatmap_v1v2v3(df_energy_control_schaefer, df_energy_clbp_schaefer)

    #atlas_schaefer = datasets.fetch_atlas_schaefer_2018(n_rois=200, yeo_networks=7, resolution_mm=1)
    #schaefer_atlas = nib.load(atlas_schaefer.maps)
    #plot_average_control(df_avg_c, atlas=schaefer_atlas, ses='v1')

    
    print(df_avg_c_con)
    plot_average_control(df_avg_c_con, df_avg_c_clbp, atlas=schaefer_atlas, ses='v1')


if __name__ == "__main__":
    main()