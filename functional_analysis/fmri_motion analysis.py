#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from nilearn import image, surface, datasets, signal, connectome, plotting, interfaces
from nilearn.maskers import MultiNiftiLabelsMasker, NiftiLabelsMasker
from nilearn.interfaces.fmriprep import load_confounds, load_confounds_strategy
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
from netneurotools import metrics, freesurfer

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_validate


def find_files_with_common_name(directory, common_name, id='func'):
    file_paths = glob.glob(os.path.join(directory, common_name))
    dict_paths = {os.path.basename(os.path.dirname(os.path.dirname(fp))) : fp for fp in file_paths}
    df = pd.DataFrame.from_dict(dict_paths, orient='index').reset_index().rename(columns={'index': 'subject', 0: 'path_'+id})
    df['img_'+id] = df['path_'+id].apply(nib.load)
    return df


def find_files_with_common_name_rapidtide(directory, common_name, id='func'):
    file_paths = glob.glob(os.path.join(directory, common_name))
    dict_paths = {os.path.basename(os.path.dirname(fp)) : fp for fp in file_paths}
    df = pd.DataFrame.from_dict(dict_paths, orient='index').reset_index().rename(columns={'index': 'subject', 0: 'path_'+id})
    df['img_'+id] = df['path_'+id].apply(nib.load)
    return df

def find_files_with_common_name_tsv(directory, common_name, id='func'):
    file_paths = glob.glob(os.path.join(directory, common_name))
    dict_paths = {os.path.basename(os.path.dirname(os.path.dirname(fp))) : fp for fp in file_paths}
    df = pd.DataFrame.from_dict(dict_paths, orient='index').reset_index().rename(columns={'index': 'subject', 0: 'path_'+id})
    df['img_'+id] = df.apply(lambda x: pd.read_csv(x['path_'+id], sep='\t', header=0, usecols=['framewise_displacement']).mean(), axis=1)
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
    slices = 575

    # Load images into dataframe
    # directories
    directory = '/media/pabaua/Transcend/fmriprep/23-10-19/V1/'
    directory_rapidtide = '/home/pabaua/dev_tpil/data/fmriprep_derivatives/rapidtide/'
    directory_first = '/home/pabaua/dev_tpil/data/fmriprep_derivatives/fsl_first/'
    directory_conn = '/home/pabaua/dev_tpil/results/results_connectflow/23-08-21_connectflow/all_schaefer/'

    # create dataframes for each files
    common_name = '*/*_task-rest_space-MNI152NLin6Asym_desc-lfofilterCleaned_bold.nii.gz'
    df_func_img = find_files_with_common_name_rapidtide(directory_rapidtide, common_name, id='func')
    common_name = '*/*_all_fast_firstseg.nii.gz'
    df_subseg_img = find_files_with_common_name_rapidtide(directory_first, common_name, id='subseg')
    common_name = '*/*/func/*_task-rest_space-MNI152NLin6Asym_desc-brain_mask.nii.gz'
    df_mask_img = find_files_with_common_name(directory, common_name, id='mask')
    common_name = '*/*/func/*_task-rest_space-MNI152NLin6Asym_desc-preproc_bold.nii.gz'
    df_func_original_img = find_files_with_common_name(directory, common_name, id='func_original')
    common_name = '*/*/func/*_task-rest_desc-confounds_timeseries.tsv'
    df_motion_img = find_files_with_common_name_tsv(directory, common_name, id='motion')
    df_dem = pd.read_excel('/home/pabaua/dev_tpil/data/Données_Paul_v2.xlsx', sheet_name=1)

    # merge dataframes to one common dataframe
    df_img = pd.merge(df_func_img, df_subseg_img, on='subject')
    df_img = pd.merge(df_img, df_mask_img, on='subject')
    df_img = pd.merge(df_img, df_dem, on='subject')
    df_img = pd.merge(df_img, df_func_original_img, on='subject')
    df_img = pd.merge(df_img, df_motion_img, on='subject')
    #df_img = df_img[df_img.subject.isin(['sub-02'])]

    # load segmentations
    target = df_img['type']
    data = df_img.drop(columns='type')
    preprocessor = ColumnTransformer([('standard_scaler', StandardScaler(), ['img_motion'])])
    model = make_pipeline(preprocessor, LogisticRegression())
    cv_results = cross_validate(model, data, target, cv=10)
    print(cv_results['test_score'].mean())




if __name__ == "__main__":
    main()
