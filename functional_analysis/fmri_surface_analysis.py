#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from nilearn import surface, datasets, signal
from nilearn.maskers import NiftiLabelsMasker
from nilearn.interfaces.fmriprep import load_confounds
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from neuromaps.datasets import fetch_fslr
from neuromaps import nulls
from neuromaps import stats as stats_neuromaps
from surfplot import Plot, utils
from scipy import stats
from scipy import ndimage
from PIL import Image
import os
import seaborn as sns
from brainspace.gradient import GradientMaps
from brainspace import utils
import brainspace.datasets
import glob
import pandas as pd
import bct.algorithms as bct_alg
import bct.utils as bct

from netneurotools.networks import match_length_degree_distribution
from netneurotools import metrics


def find_files_with_common_name(directory, common_name, id='func'):
    file_paths = glob.glob(os.path.join(directory, common_name))
    dict_paths = {os.path.basename(os.path.dirname(os.path.dirname(fp))) : fp for fp in file_paths}
    df = pd.DataFrame.from_dict(dict_paths, orient='index').reset_index().rename(columns={'index': 'subject', 0: 'path_'+id})
    df = df[~df.subject.isin(['sub-10'])]
    df['img_'+id] = df['path_'+id].apply(nib.load)
    return df

def find_files_with_common_name_rapidtide(directory, common_name, id='func'):
    file_paths = glob.glob(os.path.join(directory, common_name))
    dict_paths = {os.path.basename(os.path.dirname(fp)) : fp for fp in file_paths}
    df = pd.DataFrame.from_dict(dict_paths, orient='index').reset_index().rename(columns={'index': 'subject', 0: 'path_'+id})
    df['img_'+id] = df['path_'+id].apply(nib.load)
    return df


def find_files_with_common_name_structural(directory, common_name, id='conn'):
    file_paths = glob.glob(os.path.join(directory, common_name))
    dict_paths = {os.path.basename(os.path.dirname(os.path.dirname(fp))): fp for fp in file_paths}
    df = pd.DataFrame.from_dict(dict_paths, orient='index').reset_index().rename(columns={'index': 'participant_id', 0: 'path_'+id})
    df = df[df['participant_id'].str.contains('_ses-v1')]
    df[['subject', 'session']] = df['participant_id'].str.rsplit('_ses-', n=1, expand=True)
    df[['subject', 'num']] = df['subject'].str.rsplit('pl0', n=1, expand=True)
    df['subject'] = df['subject'] + df['num']
    df = df.drop(["participant_id", 'session', 'num'], axis=1)
    df['img_'+id] = df['path_'+id].apply(lambda x: pd.read_csv(x, header=None).values)
    return df


def compute_mean_first_passage_time(df_connectivity_matrix):
    df = df_connectivity_matrix.apply(lambda x: bct_alg.mean_first_passage_time(x))
    df = df.apply(lambda x: bct.invert(x))
    std = np.std(np.array(df.values.tolist()), axis=0)
    #df = df_connectivity_matrix.groupby('subject').apply(lambda x: bct_alg.mean_first_passage_time(bct.invert(x.drop(['subject', 'roi'], axis=1).to_numpy())))
    return df.mean(), std


def main():
    """
    Main function to execute the pipeline for connectivity mapping and visualization.
    """
    slices = 200

    # # Load images into dataframe
    # directory = '/media/pabaua/Transcend/fmriprep/23-10-19/V1/'
    # common_name = '*/*/anat/*_hemi-L_pial.surf.gii'
    # df_surf_lh_img = find_files_with_common_name(directory, common_name, id='surf_lh')
    # common_name = '*/*/anat/*_hemi-R_pial.surf.gii'
    # df_surf_rh_img = find_files_with_common_name(directory, common_name, id='surf_rh')
    # common_name = '*/*/anat/*_hemi-L_smoothwm.surf.gii'
    # df_inner_lh_img = find_files_with_common_name(directory, common_name, id='inner_lh')
    # common_name = '*/*/anat/*_hemi-R_smoothwm.surf.gii'
    # df_inner_rh_img = find_files_with_common_name(directory, common_name, id='inner_rh')
    # directory_rapidtide = '/home/pabaua/dev_tpil/data/fmriprep_derivatives/rapidtide/'
    # directory_conn = '/home/pabaua/dev_tpil/results/results_connectflow/23-08-21_connectflow/all_schaefer/'
    # common_name = '*/*_task-rest_space-MNI152NLin6Asym_desc-lfofilterCleaned_bold.nii.gz'
    # df_func_img = find_files_with_common_name_rapidtide(directory_rapidtide, common_name, id='func')
    # common_name = '*/*/func/*_task-rest_space-MNI152NLin6Asym_desc-aparcaseg_dseg.nii.gz'
    # df_seg_img = find_files_with_common_name(directory, common_name, id='seg')
    # common_name = '*/*/func/*_task-rest_space-MNI152NLin6Asym_desc-brain_mask.nii.gz'
    # df_mask_img = find_files_with_common_name(directory, common_name, id='mask')
    # df_dem = pd.read_excel('/home/pabaua/dev_tpil/data/Données_Paul_v2.xlsx', sheet_name=1)
    # df_img = pd.merge(df_func_img, df_seg_img, on='subject')
    # df_img = pd.merge(df_img, df_mask_img, on='subject')
    # df_img = pd.merge(df_img, df_dem, on='subject')
    # df_img = pd.merge(df_img, df_surf_lh_img, on='subject')
    # df_img = pd.merge(df_img, df_surf_rh_img, on='subject')
    # df_img = pd.merge(df_img, df_inner_lh_img, on='subject')
    # df_img = pd.merge(df_img, df_inner_rh_img, on='subject')
    # df_img['surf_L_comm'] = "/home/pabaua/neuromaps-data/atlases/fsLR/tpl-fsLR_den-32k_hemi-L_midthickness.surf.gii"
    # df_img['surf_R_comm'] = "/home/pabaua/neuromaps-data/atlases/fsLR/tpl-fsLR_den-32k_hemi-R_midthickness.surf.gii"
    # df_img = df_img[df_img.subject.isin(['sub-07'])]
    # print(df_img)

    # # load segmentations
    # seg_img = nib.load('/home/pabaua/dev_tpil/data/sub-07-2570132/sub-07-2570132/sub-07/anat/fsl_first/sub-07_all_fast_firstseg.nii.gz')
    # mask_img = nib.Nifti1Image((seg_img.get_fdata() == 26).astype(int), seg_img.affine, dtype=np.int32)

    # atlas = datasets.fetch_atlas_schaefer_2018(n_rois=200)
    # df_img['proj_lh'] = df_img.apply(lambda x : surface.vol_to_surf(x.img_func.slicer[:,:,:,:slices], x.surf_L_comm, mask_img=x.img_mask, kind='ball', n_samples=120).T, axis=1)                         
    # df_img['proj_rh'] = df_img.apply(lambda x : surface.vol_to_surf(x.img_func.slicer[:,:,:,:slices], x.surf_R_comm, mask_img=x.img_mask, kind='ball', n_samples=120).T, axis=1)
    # print(df_img['proj_lh'].mean().shape)
    # df_img['ext'] =  df_img.apply(lambda x : np.hstack((x.proj_lh, x.proj_rh)), axis=1)
    # df_img['ext_sub'] =  df_img.apply(lambda x : NiftiLabelsMasker(mask_img).fit_transform(x.img_func.slicer[:,:,:,:slices]), axis=1)
    # df_img['clean']=  df_img.apply(lambda x : signal.clean(x.ext, detrend=True, standardize='zscore', low_pass=0.15, high_pass=0.01, t_r=1.075, ensure_finite=True), axis=1)
    #     # Plot surfaces with functional data
    # surfaces = fetch_fslr()
    # lh, rh = surfaces['inflated']
    # plt.style.use('dark_background')
    # p = Plot(lh, rh, views=['lateral','medial', 'ventral'])
    # p.add_layer(df_img['clean'].mean()[5,:], cbar=True, cmap='jet_r')
    # fig = p.build()
    # plt.show()
    # df_img['clean_sub']=  df_img.apply(lambda x : signal.clean(x.ext_sub, detrend=True, standardize='zscore', low_pass=0.15, high_pass=0.01, t_r=1.075), axis=1)
    # #df_img['clean_tot'] =  df_img.apply(lambda x : np.vstack((x.clean, x.clean_sub)), axis=1)
    # print(df_img['clean'].mean().shape)
    # print(df_img['clean_sub'].mean().shape)
    # df_img['adj'] =  df_img.apply(lambda x : np.arctanh([(stats.pearsonr(x.clean[:,i], x.clean_sub)[0]) for i in range(x.clean.shape[1])]), axis=1)
    # print(df_img['adj'])
    # #df_img = df_img[df_img.adj.apply(lambda x: x.shape  == (107,107))]
    # # sns.heatmap(df_img.adj.mean()[:30], yticklabels=atlas.labels[:30],vmax=0.8,vmin=-0.8)
    # # plt.show()

    # labelling_schaefer = brainspace.datasets.load_parcellation('schaefer', scale=200, join=True)
    # mask = (labelling_schaefer != 0)
    # df_img['surf_data_func'] = df_img.apply(lambda x: utils.parcellation.map_to_labels(x.adj[-1, :-1], labelling_schaefer, mask=mask, fill=np.nan), axis=1)
    # df_img['rotated'] = df_img.apply(lambda x: nulls.alexander_bloch(x.surf_data_func, atlas='fsLR', density='32k', n_perm=100, seed=1234), axis=1)
    # # Plot surfaces with functional data
    # surfaces = fetch_fslr()
    # lh, rh = surfaces['inflated']
    # p = Plot(lh, rh, views=['lateral','medial'])
    # p.add_layer(df_img['surf_data_func'].mean(), cbar=True, cmap='inferno')
    # fig = p.build()
    # plt.show()

    # id='sc'
    # common_name = '*/Compute_Connectivity/sc_vol_normalized.csv'
    # df_conn_img = find_files_with_common_name_structural(directory_conn, common_name, id=id)
    # df_img = pd.merge(df_img, df_conn_img, on='subject')

    # id='commit2'
    # common_name = '*/Compute_Connectivity/commit2_weights.csv'
    # df_conn_img = find_files_with_common_name_structural(directory_conn, common_name, id=id)
    # df_img = pd.merge(df_img, df_conn_img, on='subject')

    # id='afd'
    # common_name = '*/Compute_Connectivity/afd_fixel.csv'
    # df_conn_img = find_files_with_common_name_structural(directory_conn, common_name, id=id)
    # df_img = pd.merge(df_img, df_conn_img, on='subject')

    # id='len'
    # common_name = '*/Compute_Connectivity/len.csv'
    # df_conn_img = find_files_with_common_name_structural(directory_conn, common_name, id=id)
    # df_img = pd.merge(df_img, df_conn_img, on='subject')
    # print(df_img)

    # list_metrics = []
    # for id in ['sc', 'commit2', 'len']:
    #     # shortest path
    #     metric = '_sp'
    #     df_img[id + metric] = df_img.apply(lambda x: bct.invert(bct_alg.distance_wei(bct.invert(x['img_' + id]))[0]), axis=1)
    #     df_img[id + metric] = df_img.apply(lambda x: utils.parcellation.map_to_labels(x[id + metric][7,15:], labelling_schaefer, mask=mask, fill=np.nan), axis=1)
    #     # Plot surfaces with functional data
    #     surfaces = fetch_fslr()
    #     lh, rh = surfaces['inflated']
    #     p = Plot(lh, rh, views=['lateral','medial'])
    #     p.add_layer(df_img[id + metric].mean(), cbar=True, cmap='inferno')
    #     fig = p.build()
    #     plt.show()
    #     df_img[id + metric] = df_img.apply(lambda x: stats_neuromaps.compare_images(x.surf_data_func, x[id + metric], nulls=x.rotated)[0], axis=1)
    #     list_metrics += [id + metric]

    #     # mean first passage time
    #     metric = '_mfpt'
    #     df_img[id + metric] = df_img.apply(lambda x: bct.invert(bct_alg.mean_first_passage_time(x['img_' + id])), axis=1)
    #     df_img[id + metric] = df_img.apply(lambda x: utils.parcellation.map_to_labels(x[id + metric][7,15:], labelling_schaefer, mask=mask, fill=np.nan), axis=1)
    #     df_img[id + metric] = df_img.apply(lambda x: stats_neuromaps.compare_images(x.surf_data_func, x[id + metric], nulls=x.rotated)[0], axis=1)
    #     list_metrics += [id + metric]

    #     # mean first passage time
    #     metric = '_si'
    #     df_img[id + metric] = df_img.apply(lambda x: bct_alg.search_information(x['img_' + id]), axis=1)
    #     df_img[id + metric] = df_img.apply(lambda x: utils.parcellation.map_to_labels(x[id + metric][7,15:], labelling_schaefer, mask=mask, fill=np.nan), axis=1)
    #     df_img[id + metric] = df_img.apply(lambda x: stats_neuromaps.compare_images(x.surf_data_func, x[id + metric], nulls=x.rotated)[0], axis=1)
    #     list_metrics += [id + metric]

    # #df_img.to_pickle('/home/pabaua/dev_tpil/data/correlations_ohbm_compcorr.pkl')
    df_img = pd.read_pickle('/home/pabaua/dev_tpil/data/correlations_ohbm_compcorr.pkl')
    print(df_img)
    #df_img = df_img.melt(var_name='Category', value_name="Pearson's r")
    # Select only numeric columns for the y-axis and a categorical column for the hue
    numeric_columns = df_img.select_dtypes(include='float').columns
    hue_column = 'type'  # Replace with your actual hue column name
    # Melt the DataFrame to long format for easier plotting with seaborn
    long_df = df_img.melt(id_vars=hue_column, value_vars=numeric_columns, var_name='features', value_name='value')

    # Create the violin plot with hue
    sns.set_theme(style="whitegrid", font_scale=1.1)
    sns.violinplot(data=long_df, x='features', y='value', hue=hue_column, split=True, gap=0.1, inner="quart") 
    # sns.violinplot(data=df_img, hue='type', split=True)
    plt.ylim((-0.7,0.7))
    plt.ylabel("pearson's r")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()
    # id='commit2'
    # common_name = '*/Compute_Connectivity/commit2_weights.csv'
    # df_conn_img = find_files_with_common_name_structural(directory_conn, common_name, id='commit2')
    # df_img = pd.merge(df_img, df_conn_img, on='subject')



    # for id in ['sc', 'commit2']:
    #     labelling_schaefer = brainspace.datasets.load_parcellation('schaefer', scale=200, join=True)
    #     mask = (labelling_schaefer != 0)
    #     surf_data_func = utils.parcellation.map_to_labels(np.abs(df_img.adj.mean()[-1, :-1]), labelling_schaefer, mask=mask, fill=np.nan)
    #     rotated = nulls.alexander_bloch(surf_data_func, atlas='fsLR', density='32k', n_perm=100, seed=1234)

    #     df_img[id + 'sp'] = df_img.apply(lambda x: bct.invert(bct_alg.distance_wei(bct.invert(x['img_' + id]))[0]), axis=1)
    #     surf_data_sp = utils.parcellation.map_to_labels(np.abs(df_img[id + 'sp'].mean()[7,15:]), labelling_schaefer, mask=mask, fill=np.nan)
    #     corr_sp, pval_sp, nulls_sp = stats_neuromaps.compare_images(surf_data_func, surf_data_sp, nulls=rotated, return_nulls=True)
    #     print(f'Correlation: r = {corr_sp:.02f}, p = {pval_sp:.04f}')

    #     df_img[id + 'mfpt'] = df_img.apply(lambda x: bct.invert(bct_alg.mean_first_passage_time(x['img_' + id])), axis=1)
    #     surf_data_mfpt = utils.parcellation.map_to_labels(np.abs(df_img[id + 'mfpt'].mean()[7,15:]), labelling_schaefer, mask=mask, fill=np.nan)
    #     corr_mfpt, pval_mfpt, nulls_mfpt = stats_neuromaps.compare_images(surf_data_func, surf_data_mfpt, nulls=rotated, return_nulls=True)
    #     print(f'Correlation: r = {corr_mfpt:.02f}, p = {pval_mfpt:.04f}')

    #     df_img[id + 'si'] = df_img.apply(lambda x: bct_alg.search_information(x['img_' + id]), axis=1)
    #     surf_data_si = utils.parcellation.map_to_labels(np.abs(df_img[id + 'si'].mean()[7,15:]), labelling_schaefer, mask=mask, fill=np.nan)
    #     corr_si, pval_si, nulls_si = stats_neuromaps.compare_images(surf_data_func, surf_data_si, nulls=rotated, return_nulls=True)
    #     print(f'Correlation: r = {corr_si:.02f}, p = {pval_si:.04f}')
    #     dict = {id: [corr_sp, corr_mfpt, corr_si]}



    # communication_names = ['mfpt', 'shortest path', 'search information']
    # plt.figure(figsize = (7,7))
    # sns.set(style="ticks",font_scale=2)
    # p1 = sns.boxplot([nulls_mfpt, nulls_sp, nulls_si], color='white', orient='h')
    # ax = plt.gca()
    # ax.set_xlim([-0.2, 0.2])
    # ax.set_yticklabels(communication_names)
    # plt.scatter(y=range(len(communication_names)), x=[corr_mfpt, corr_sp, corr_si], s=50)
    # plt.legend(frameon=False)
    # plt.xlabel("pearson's r")
    # plt.ylabel("communication model")
    # plt.legend(frameon=False)
    # p1.xaxis.tick_top()
    # plt.tight_layout()
    # plt.show()






    # # Plot surfaces with functional data
    # surfaces = fetch_fslr()
    # lh, rh = surfaces['inflated']
    # p = Plot(lh, rh, views=['lateral','medial'])
    # p.add_layer(surf_data_func, cbar=True, cmap='inferno')
    # fig = p.build()
    # plt.show()


    # # Plot surfaces with functional data
    # surfaces = fetch_fslr()
    # lh, rh = surfaces['inflated']
    # p = Plot(lh, rh, views=['lateral','medial'])
    # p.add_layer(surf_data_mfpt, cbar=True, cmap='inferno')
    # fig = p.build()
    # plt.show()






    #mask_img = nib.Nifti1Image((seg_img.get_fdata() == 26).astype(int), seg_img.affine, dtype=np.int32)
    #ext_sub = NiftiLabelsMasker(seg_img).fit_transform(func_img.slicer[:,:,:,:slices])

    # # Define surfaces
    # inner_lh = f'{FS_PATH}/sub-pl007_ses-v1.L.white.32k_fs_LR.surf.gii'
    # inner_rh = f'{FS_PATH}/sub-pl007_ses-v1.R.white.32k_fs_LR.surf.gii'
    # surf_lh = f'{FS_PATH}/sub-pl007_ses-v1.L.pial.32k_fs_LR.surf.gii'
    # surf_rh = f'{FS_PATH}/sub-pl007_ses-v1.R.pial.32k_fs_LR.surf.gii'

    # # Process functional image and surface data
    # slices = func_img.get_fdata().shape[3]
    # #slices = 34
    # print(slices)

    # # Fetch surface atlas
    # atlas = nib.load('/home/pabaua/dev_tpil/data/BN/BN_Atlas_for_FSL/Brainnetome/BNA-maxprob-thr0-1mm.nii.gz')
    # atlas = datasets.fetch_atlas_schaefer_2018()
    # ext_cortex = NiftiLabelsMasker(atlas.maps, labels=atlas.labels).fit_transform(func_img.slicer[:,:,:,:slices])
    # #mask_img = nib.Nifti1Image((seg_img.get_fdata() == 26).astype(int), seg_img.affine, dtype=np.int32)
    # ext_sub = NiftiLabelsMasker(seg_img).fit_transform(func_img.slicer[:,:,:,:slices])

    # clean_cortex = signal.clean(ext_cortex, detrend=True, standardize='zscore_sample', low_pass=0.15, high_pass=0.01, t_r=1.075).T
    # clean_sub = signal.clean(ext_sub, detrend=True, standardize='zscore_sample', low_pass=0.15, high_pass=0.01, t_r=1.075).T
    # clean_parcels = np.vstack((clean_cortex, clean_sub))
    # print(clean_parcels.shape)

    # correlation_measure = connectome.ConnectivityMeasure(kind='correlation')
    # correlation_matrix = correlation_measure.fit_transform([clean_parcels.T])[0]
    # sns.heatmap(correlation_matrix)
    # plt.show()

    # # labelling_schaefer = brainspace.datasets.load_parcellation('schaefer', scale=400, join=True)
    # # mask = labelling_schaefer != 0
    # # surf_data = utils.parcellation.map_to_labels(np.abs(correlation_matrix[-1,:]), labelling_schaefer, mask=mask, fill=np.nan)

    # # # Plot surfaces with functional data
    # # surfaces = fetch_fslr()
    # # lh, rh = surfaces['inflated']
    # # p = Plot(lh, rh, views=['lateral','medial'])
    # # p.add_layer(surf_data, cbar=True, cmap='inferno')
    # # fig = p.build()
    # # plt.show()




    # gm = GradientMaps(n_components=10, random_state=0)
    # gm.fit(correlation_matrix)
    # print(gm.gradients_[:,0].shape)

    # labelling_lh = nib.load('/home/pabaua/dev_tpil/data/BN/BN_Atlas_freesurfer/fsaverage/fsaverage_LR32k/fsaverage.L.BN_Atlas.32k_fs_LR.label.gii').agg_data()
    # labelling_rh = nib.load('/home/pabaua/dev_tpil/data/BN/BN_Atlas_freesurfer/fsaverage/fsaverage_LR32k/fsaverage.R.BN_Atlas.32k_fs_LR.label.gii').agg_data()
    # labelling_schaefer = brainspace.datasets.load_parcellation('schaefer', scale=400, join=True)
    # grad = [None] * 2
    # mask = labelling_schaefer != 0
    # for i in range(2):
    #     # map the gradient to the parcels
    #     grad[i] = utils.parcellation.map_to_labels(gm.gradients_[:, i], labelling_schaefer, mask=mask, fill=np.nan)


    # # Plot surfaces with functional data
    # surfaces = fetch_fslr()
    # lh, rh = surfaces['inflated']
    # p = Plot(lh, rh, views=['lateral','medial'])
    # p.add_layer(grad[0], cbar=True, cmap='inferno')
    # fig = p.build()
    # plt.show()
    # # Plot surfaces with functional data
    # surfaces = fetch_fslr()
    # lh, rh = surfaces['inflated']
    # p = Plot(lh, rh, views=['lateral','medial'])
    # p.add_layer(grad[1], cbar=True, cmap='inferno')
    # fig = p.build()
    # plt.show()


    # fig, ax = plt.subplots(1, figsize=(5, 4))
    # ax.scatter(range(gm.lambdas_.size), gm.lambdas_)
    # ax.set_xlabel('Component Nb')
    # ax.set_ylabel('Eigenvalue')
    # plt.show()

    # # # Compute correlation
    # # stat_map_lh = np.abs([stats.pearsonr(clean_sub, clean_lh[:,i])[0] for i in range(clean_lh.shape[1])])
    # # stat_map_rh = np.abs([stats.pearsonr(clean_sub, clean_rh[:,i])[0] for i in range(clean_rh.shape[1])])

    # # Plot surfaces with correlation data
    # #p = Plot(infl_lh, infl_rh, views=['lateral','medial'], zoom=1.2)
    # #stat_map_lh = utils.threshold(stat_map_lh, 0.3)
    # #stat_map_rh = utils.threshold(stat_map_rh, 0.3)
    # # surfaces = fetch_fslr()
    # # lh, rh = surfaces['inflated']
    # # p = Plot(lh, rh, views=['lateral','medial'])
    # # p.add_layer({'left':stat_map_lh, 'right':stat_map_rh}, cbar=True, cmap='inferno')
    # # fig = p.build()
    # # plt.show(block=True)

    # # surfaces = fetch_fslr()
    # # lh, rh = surfaces['inflated']
    # # p = Plot(lh, rh, views=['lateral','medial'])
    # # lh_BN = nib.load('/home/pabaua/dev_tpil/data/BN/BN_Atlas_freesurfer/fsaverage/fsaverage_LR32k/fsaverage.L.BN_Atlas.32k_fs_LR.label.gii').agg_data()
    # # rh_BN = nib.load('/home/pabaua/dev_tpil/data/BN/BN_Atlas_freesurfer/fsaverage/fsaverage_LR32k/fsaverage.R.BN_Atlas.32k_fs_LR.label.gii').agg_data() 
    # # val_lh = np.zeros(stat_map_lh.shape[0])
    # # val_rh = np.zeros(stat_map_rh.shape[0])
    # # for i in np.arange(211):
    # #     val_lh += np.where(lh_BN == i, np.mean((lh_BN == i) * stat_map_lh), 0)
    # #     val_rh += np.where(rh_BN == i, np.mean((rh_BN == i) * stat_map_rh), 0)
    # # p.add_layer({'left':val_lh, 'right':val_rh}, cbar=True, cmap='inferno')
    # # fig = p.build()
    # # plt.show(block=True)



    # # clean_lh = ndimage.gaussian_filter1d(clean_lh, axis=0, sigma=3)
    # # clean_rh = ndimage.gaussian_filter1d(clean_rh, axis=0, sigma=3)

    # # # Save each frame as an image
    # # frames = []
    # # for frame in range(surf_data.shape[0]):
    # #     p = Plot(infl_lh, infl_rh, views=['lateral', 'medial'], flip=True)
    # #     p.add_layer({'left':clean_lh[frame,:], 'right':clean_rh[frame,:]}, cbar=True, cmap='inferno', color_range=(-3,3), cbar_label=f"t = {np.round(frame* 1.075)} s")
    # #     fig = p.build()
    # #     filename = f"frame_{frame}.png"
    # #     plt.savefig(filename)
    # #     frames.append(filename)
    # #     plt.close(fig)

    # # # Create a GIF from the saved frames
    # # with Image.open(frames[0]) as img:
    # #     img.save("animation.gif", save_all=True, append_images=[Image.open(f) for f in frames[1:]], duration=100, loop=0)

    # # # Remove the individual frame images
    # # for frame in frames:
    # #     os.remove(frame)





if __name__ == "__main__":
    main()
