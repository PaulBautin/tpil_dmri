#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from nilearn import image, surface, datasets, signal, connectome, plotting, interfaces
from nilearn.maskers import MultiNiftiLabelsMasker, NiftiLabelsMasker
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from neuromaps.datasets import fetch_fslr
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
    df = df[~df.subject.isin(['sub-10'])]
    df['img_'+id] = df['path_'+id].apply(nib.load)
    return df

def main():
    """
    Main function to execute the pipeline for connectivity mapping and visualization.
    """
    slices = 575

    # Load images into dataframe
    directory_rapidtide = '/home/pabaua/dev_tpil/data/rapidtide/'
    common_name = '*/*_task-rest_space-MNI152NLin6Asym_desc-lfofilterCleaned_bold.nii.gz'
    df_func_img = find_files_with_common_name_rapidtide(directory_rapidtide, common_name, id='func')
    common_name = '*/*_task-rest_space-MNI152NLin6Asym_desc-MTT_map.nii.gz'
    #common_name = '*/*task-rest_space-MNI152NLin6Asym_desc-maxtime_map.nii.gz'
    df_mtt_img = find_files_with_common_name_rapidtide(directory_rapidtide, common_name, id='mtt')
    df_dem = pd.read_excel('/home/pabaua/dev_tpil/data/Données_Paul_v2.xlsx', sheet_name=1)
    df_img = pd.merge(df_func_img, df_mtt_img, on='subject')
    df_img = pd.merge(df_img, df_dem, on='subject')
    print(df_img)

    # plot 
    ref = nib.load('/home/pabaua/dev_tpil/data/rapidtide/sub-02/sub-02_task-rest_space-MNI152NLin6Asym_desc-mean_map.nii.gz')
    df_img['mtt_data'] = df_img.apply(lambda x : x.img_mtt.get_fdata(), axis=1)
    df_img['mtt_data'] = df_img.apply(lambda x : np.where((x.mtt_data >= -10) & (x.mtt_data <= 10), x.mtt_data, np.nan), axis=1)
    mean_mtt_clbp = df_img[(df_img['type'] == 'DC')]['mtt_data'].mean(skipna=True)
    mean_mtt_con = df_img[(df_img['type'] == 'Sain')]['mtt_data'].mean(skipna=True)
    #zscore_mtt = (mean_mtt_clbp - mean_mtt_con) / df_img[df_img['type'] == 'Sain']['mtt_data'].values.std()
    zscore_mtt = (mean_mtt_clbp - mean_mtt_con)
    #zscore_mtt = np.where(zscore_mtt >= 1, 1, 0)
    mean_mtt_img_clbp = nib.Nifti1Image(mean_mtt_clbp, ref.affine, dtype=np.int32)
    mean_mtt_img_con = nib.Nifti1Image(mean_mtt_con, ref.affine, dtype=np.int32)
    zscore_mtt_img = nib.Nifti1Image(zscore_mtt, ref.affine, dtype=np.int32)
    fig, axes = plt.subplots(3, 1, figsize=(19, 16))
    plotting.plot_stat_map(mean_mtt_img_clbp, vmax=2, axes=axes[0], title='Mean Time Travel CLBP',cut_coords=(0, -20, 23))
    plotting.plot_stat_map(mean_mtt_img_con, vmax=2, axes=axes[1], title='Mean Time Travel control',cut_coords=(0, -20, 23))
    plotting.plot_stat_map(zscore_mtt_img, vmax=2, axes=axes[2], title='zscore meam Time Travel ',cut_coords=(0, -20, 23))
    plt.show() 



if __name__ == "__main__":
    main()
