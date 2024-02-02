#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import nibabel as nib
import numpy as np
from nilearn import signal, image, surface
from nilearn.maskers import NiftiLabelsMasker
from nilearn.interfaces.fmriprep import load_confounds, load_confounds_strategy
from scipy import stats
from surfplot import Plot, utils
from neuromaps.datasets import fetch_fslr
import matplotlib.pyplot as plt


def save_cifti_file(data, cifti_file, cifti_output):
    """
    Save streamline incidence data into a new CIFTI file.

    Parameters
    ----------
    streamline_incidence : ndarray
        The streamline incidence data.
    cifti_file : str
        Path to the original CIFTI file.
    cifti_output : str
        Path to save the new CIFTI file.

    Returns
    -------
    new_img : nib.Cifti2Image
        The new CIFTI image object.
    """
    # Load original CIFTI file once to improve efficiency
    original_cifti = nib.load(cifti_file)
    original_data = original_cifti.get_fdata()
    # Apply the streamline incidence to the original data
    new_data = data[np.newaxis, ...] * np.ones_like(original_data)
    # Create a new CIFTI image object with the updated data
    new_img = nib.Cifti2Image(new_data, header=original_cifti.header, nifti_header=original_cifti.nifti_header)
    # Save the new CIFTI image to disk
    new_img.to_filename(cifti_output)
    return new_img


def plot_surface(stat_map_lh, stat_map_rh):
    """
    Plot a surface representation of a CIFTI file.

    Parameters
    ----------
    cifti_file : str or nib.Cifti2Image
        The CIFTI file or image object to plot.
    """
    surfaces = fetch_fslr()
    plt.style.use('dark_background')
    lh, rh = surfaces['inflated']
    p = Plot(lh, rh, views=['lateral','medial', 'ventral'], zoom=1.2)
    #stat_map_lh = utils.threshold(stat_map_lh, 0.15)
    #stat_map_rh = utils.threshold(stat_map_rh, 0.15)
    p.add_layer({'left':stat_map_lh, 'right':stat_map_rh}, cbar=True, cmap='jet')
    fig = p.build()
    plt.show(block=True)


def load_dtseries_file(dtseries_file_path):
    """
    Load a CIFTI dtseries file using Nibabel.
    
    :param dtseries_file_path: Path to the CIFTI dtseries file.
    :return: CIFTI data array.
    """
    dtseries_img = nib.load(dtseries_file_path)
    dtseries_data = dtseries_img.get_fdata()
    return dtseries_img, dtseries_data

def identify_nucleus_accumbens(dtseries_img, accumbens_label):
    """
    Identify the nucleus accumbens region in the CIFTI file.
    
    :param dtseries_img: CIFTI image.
    :param accumbens_label: Label corresponding to the nucleus accumbens.
    :return: Boolean mask for the nucleus accumbens.
    """
    # Access the brain model information
    brain_models = list(dtseries_img.header.get_index_map(1).brain_models)

    # Iterate through the brain models to find the nucleus accumbens
    # Create a dictionary for storing the number of voxel indices for each brain structure
    brain_structure_indices = {bm.brain_structure: idx for idx, bm in enumerate(brain_models)}
    i = brain_structure_indices[accumbens_label]
    accumbens_indices = range(brain_models[i].index_offset, brain_models[i].index_offset + brain_models[i].index_count)
    return accumbens_indices

def compute_correlations(dtseries_data, seed_time_series):
    """
    Compute correlations using Nilearn.
    
    :param dtseries_data: CIFTI data array.
    :param seed_time_series: Time series of the seed region (nucleus accumbens).
    :return: Correlation map.
    """
    path_func_original = '/home/pabaua/dev_tpil/results/results_fmriprep/24-01-16_fmriprep/sub-007/ses-v1/func/sub-007_ses-v1_task-rest_space-T1w_desc-preproc_bold.nii.gz'
    confounds_simple, sample_mask = load_confounds_strategy(path_func_original, denoise_strategy='scrubbing')
    print(confounds_simple)
    cleaned_data = signal.clean(dtseries_data, detrend=True, standardize=True, low_pass=0.15, high_pass=0.01, t_r=1.075, confounds=confounds_simple, ensure_finite=True)
    seed_time_series = signal.clean(seed_time_series, detrend=True, standardize=True, low_pass=0.15, high_pass=0.01, t_r=1.075, confounds=confounds_simple, ensure_finite=True)
    print(cleaned_data)
    print(seed_time_series)
   
    stat_map = np.zeros(cleaned_data.T.shape[0])
    for i in range(cleaned_data.T.shape[0]):
        stat_map[i] = np.abs(stats.pearsonr(seed_time_series[:,0], cleaned_data[:,i])[0])
    print(stat_map)
    return stat_map



def main():
    """
    Main function to execute the pipeline for connectivity mapping and visualization.
    """
    # Example usage
    #dtseries_file_path = '/home/pabaua/dev_tpil/results/results_fmriprep/24-01-17_fmriprep_multiecho/sub-001/func/sub-001_task-rest_space-fsLR_den-91k_bold.dtseries.nii'
    bold_img = nib.load('/home/pabaua/dev_tpil/results/results_fmriprep/24-01-16_fmriprep/sub-007/ses-v1/func/sub-007_ses-v1_task-rest_space-T1w_desc-preproc_bold.nii.gz')
    lh_surf_white = '/home/pabaua/dev_tpil/results/results_article/results/sub-007/ses-v1/Fs_ciftify/sub-007_ses-v1_white_resampled_lh.surf.gii'
    rh_surf_white = '/home/pabaua/dev_tpil/results/results_article/results/sub-007/ses-v1/Fs_ciftify/sub-007_ses-v1_white_resampled_rh.surf.gii'
    lh_surf_pial = '/home/pabaua/dev_tpil/results/results_article/results/sub-007/ses-v1/Fs_ciftify/sub-007_ses-v1_pial_resampled_lh.surf.gii'
    rh_surf_pial = '/home/pabaua/dev_tpil/results/results_article/results/sub-007/ses-v1/Fs_ciftify/sub-007_ses-v1_pial_resampled_rh.surf.gii'
    mask_img = nib.load('/home/pabaua/dev_tpil/results/results_fmriprep/24-01-16_fmriprep/sub-007/ses-v1/func/sub-007_ses-v1_task-rest_space-T1w_desc-brain_mask.nii.gz')

    # Load dtseries file
    #dtseries_img, dtseries_data = load_dtseries_file(dtseries_file_path)
    dtseries_data_lh = surface.vol_to_surf(bold_img, lh_surf_pial, inner_mesh=lh_surf_white, mask_img=mask_img).T
    #dtseries_data_lh[np.isnan(dtseries_data_lh)] = 0.0
    dtseries_data_rh = surface.vol_to_surf(bold_img, rh_surf_pial, inner_mesh=rh_surf_white, mask_img=mask_img).T
    #dtseries_data_rh[np.isnan(dtseries_data_rh)] = 0.0
    #print(dtseries_data.shape)
    
    # Identify nucleus accumbens and extract time series
    # This step needs the correct label and method for your specific dataset
    #nucleus_accumbens_mask = identify_nucleus_accumbens(dtseries_img, 'CIFTI_STRUCTURE_ACCUMBENS_LEFT')
    #seed_time_series = np.mean(dtseries_data[:, nucleus_accumbens_mask], axis=1)
    # load segmentations
    seg_img = nib.load('/home/pabaua/dev_tpil/results/results_article/results/sub-007/ses-v1/Subcortex_segmentation/sub-007_ses-v1_all_fast_firstseg.nii.gz')
    mask_img = nib.Nifti1Image((seg_img.get_fdata() == 26).astype(int), seg_img.affine, dtype=np.int32)
    seed_time_series = NiftiLabelsMasker(mask_img).fit_transform(bold_img)
    #seed_time_series[np.isnan(seed_time_series)] = 0.0
    print(seed_time_series)


    # Compute correlations using Nilearn
    correlation_map_lh = compute_correlations(dtseries_data_lh, seed_time_series)
    correlation_map_rh = compute_correlations(dtseries_data_rh, seed_time_series)
    print(correlation_map_lh)
    print(correlation_map_lh.shape)

    # Process the correlation map further as needed
    #cifti_output = '/home/pabaua/dev_tpil/results/cifti_output.dtseries.nii'
    #img = save_cifti_file(correlation_map, dtseries_file_path, cifti_output)
    plot_surface(correlation_map_lh, correlation_map_rh)


if __name__ == "__main__":
    main()
