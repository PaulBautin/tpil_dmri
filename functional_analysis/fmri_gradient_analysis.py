#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from nilearn import image, surface, datasets, signal, connectome
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

# Constants for file path prefixes
BASE_PATH = '/home/pabaua/dev_tpil/data/sub-07-2570132/sub-07-2570132/sub-07'
ANAT_PATH = f'{BASE_PATH}/anat'
FUNC_PATH = f'{BASE_PATH}/func'
FS_PATH = '/home/pabaua/dev_tpil/results/results_new_bundle/23-10-19_accumbofrontal/results/sub-pl007_ses-v1/Fs_ciftify/sub-pl007_ses-v1/MNINonLinear/fsaverage_LR32k'

def find_files_with_common_name(directory, common_name, labels_list=None):
    file_paths = glob.glob(directory + common_name)
    n = range(len(file_paths))
    dict_paths = {os.path.basename(os.path.dirname(os.path.dirname(file_paths[i]))) : file_paths[i] for i in range(len(file_paths))}
    return dict_paths

def main():
    """
    Main function to execute the pipeline for connectivity mapping and visualization.
    """
    
    # Load images
    directory = '/media/pabaua/Transcend/fmriprep/23-10-19/V1/'
    common_name = '*/*/func/*_task-rest_space-MNI152NLin6Asym_desc-preproc_bold.nii.gz'
    func_img = find_files_with_common_name(directory, common_name, labels_list=None)
    func_img = nib.load(f'{FUNC_PATH}/sub-07_task-rest_space-MNI152NLin6Asym_desc-preproc_bold.nii.gz')
    seg_img = nib.load(f'{FUNC_PATH}/sub-07_task-rest_space-MNI152NLin6Asym_desc-aseg_dseg.nii.gz')
    seg_img = nib.load(f'{ANAT_PATH}/fsl_first/sub-07_all_fast_firstseg.nii.gz')

    # Define surfaces
    inner_lh = f'{FS_PATH}/sub-pl007_ses-v1.L.white.32k_fs_LR.surf.gii'
    inner_rh = f'{FS_PATH}/sub-pl007_ses-v1.R.white.32k_fs_LR.surf.gii'
    surf_lh = f'{FS_PATH}/sub-pl007_ses-v1.L.pial.32k_fs_LR.surf.gii'
    surf_rh = f'{FS_PATH}/sub-pl007_ses-v1.R.pial.32k_fs_LR.surf.gii'

    # Process functional image and surface data
    slices = func_img.get_fdata().shape[3]
    #slices = 34
    print(slices)

    # Fetch surface atlas
    atlas = nib.load('/home/pabaua/dev_tpil/data/BN/BN_Atlas_for_FSL/Brainnetome/BNA-maxprob-thr0-1mm.nii.gz')
    atlas = datasets.fetch_atlas_schaefer_2018()
    ext_cortex = NiftiLabelsMasker(atlas.maps, labels=atlas.labels).fit_transform(func_img.slicer[:,:,:,:slices])
    #mask_img = nib.Nifti1Image((seg_img.get_fdata() == 26).astype(int), seg_img.affine, dtype=np.int32)
    ext_sub = NiftiLabelsMasker(seg_img).fit_transform(func_img.slicer[:,:,:,:slices])

    clean_cortex = signal.clean(ext_cortex, detrend=True, standardize='zscore_sample', low_pass=0.15, high_pass=0.01, t_r=1.075).T
    clean_sub = signal.clean(ext_sub, detrend=True, standardize='zscore_sample', low_pass=0.15, high_pass=0.01, t_r=1.075).T
    clean_parcels = np.vstack((clean_cortex, clean_sub))
    print(clean_parcels.shape)

    correlation_measure = connectome.ConnectivityMeasure(kind='correlation')
    correlation_matrix = correlation_measure.fit_transform([clean_parcels.T])[0]
    sns.heatmap(correlation_matrix)
    plt.show()

    # labelling_schaefer = brainspace.datasets.load_parcellation('schaefer', scale=400, join=True)
    # mask = labelling_schaefer != 0
    # surf_data = utils.parcellation.map_to_labels(np.abs(correlation_matrix[-1,:]), labelling_schaefer, mask=mask, fill=np.nan)

    # # Plot surfaces with functional data
    # surfaces = fetch_fslr()
    # lh, rh = surfaces['inflated']
    # p = Plot(lh, rh, views=['lateral','medial'])
    # p.add_layer(surf_data, cbar=True, cmap='inferno')
    # fig = p.build()
    # plt.show()




    gm = GradientMaps(n_components=10, random_state=0)
    gm.fit(correlation_matrix)
    print(gm.gradients_[:,0].shape)

    labelling_lh = nib.load('/home/pabaua/dev_tpil/data/BN/BN_Atlas_freesurfer/fsaverage/fsaverage_LR32k/fsaverage.L.BN_Atlas.32k_fs_LR.label.gii').agg_data()
    labelling_rh = nib.load('/home/pabaua/dev_tpil/data/BN/BN_Atlas_freesurfer/fsaverage/fsaverage_LR32k/fsaverage.R.BN_Atlas.32k_fs_LR.label.gii').agg_data()
    labelling_schaefer = brainspace.datasets.load_parcellation('schaefer', scale=400, join=True)
    grad = [None] * 2
    mask = labelling_schaefer != 0
    for i in range(2):
        # map the gradient to the parcels
        grad[i] = utils.parcellation.map_to_labels(gm.gradients_[:, i], labelling_schaefer, mask=mask, fill=np.nan)


    # Plot surfaces with functional data
    surfaces = fetch_fslr()
    lh, rh = surfaces['inflated']
    p = Plot(lh, rh, views=['lateral','medial'])
    p.add_layer(grad[0], cbar=True, cmap='inferno')
    fig = p.build()
    plt.show()
    # Plot surfaces with functional data
    surfaces = fetch_fslr()
    lh, rh = surfaces['inflated']
    p = Plot(lh, rh, views=['lateral','medial'])
    p.add_layer(grad[1], cbar=True, cmap='inferno')
    fig = p.build()
    plt.show()


    fig, ax = plt.subplots(1, figsize=(5, 4))
    ax.scatter(range(gm.lambdas_.size), gm.lambdas_)
    ax.set_xlabel('Component Nb')
    ax.set_ylabel('Eigenvalue')
    plt.show()

    # # Compute correlation
    # stat_map_lh = np.abs([stats.pearsonr(clean_sub, clean_lh[:,i])[0] for i in range(clean_lh.shape[1])])
    # stat_map_rh = np.abs([stats.pearsonr(clean_sub, clean_rh[:,i])[0] for i in range(clean_rh.shape[1])])

    # Plot surfaces with correlation data
    #p = Plot(infl_lh, infl_rh, views=['lateral','medial'], zoom=1.2)
    #stat_map_lh = utils.threshold(stat_map_lh, 0.3)
    #stat_map_rh = utils.threshold(stat_map_rh, 0.3)
    # surfaces = fetch_fslr()
    # lh, rh = surfaces['inflated']
    # p = Plot(lh, rh, views=['lateral','medial'])
    # p.add_layer({'left':stat_map_lh, 'right':stat_map_rh}, cbar=True, cmap='inferno')
    # fig = p.build()
    # plt.show(block=True)

    # surfaces = fetch_fslr()
    # lh, rh = surfaces['inflated']
    # p = Plot(lh, rh, views=['lateral','medial'])
    # lh_BN = nib.load('/home/pabaua/dev_tpil/data/BN/BN_Atlas_freesurfer/fsaverage/fsaverage_LR32k/fsaverage.L.BN_Atlas.32k_fs_LR.label.gii').agg_data()
    # rh_BN = nib.load('/home/pabaua/dev_tpil/data/BN/BN_Atlas_freesurfer/fsaverage/fsaverage_LR32k/fsaverage.R.BN_Atlas.32k_fs_LR.label.gii').agg_data() 
    # val_lh = np.zeros(stat_map_lh.shape[0])
    # val_rh = np.zeros(stat_map_rh.shape[0])
    # for i in np.arange(211):
    #     val_lh += np.where(lh_BN == i, np.mean((lh_BN == i) * stat_map_lh), 0)
    #     val_rh += np.where(rh_BN == i, np.mean((rh_BN == i) * stat_map_rh), 0)
    # p.add_layer({'left':val_lh, 'right':val_rh}, cbar=True, cmap='inferno')
    # fig = p.build()
    # plt.show(block=True)



    # clean_lh = ndimage.gaussian_filter1d(clean_lh, axis=0, sigma=3)
    # clean_rh = ndimage.gaussian_filter1d(clean_rh, axis=0, sigma=3)

    # # Save each frame as an image
    # frames = []
    # for frame in range(surf_data.shape[0]):
    #     p = Plot(infl_lh, infl_rh, views=['lateral', 'medial'], flip=True)
    #     p.add_layer({'left':clean_lh[frame,:], 'right':clean_rh[frame,:]}, cbar=True, cmap='inferno', color_range=(-3,3), cbar_label=f"t = {np.round(frame* 1.075)} s")
    #     fig = p.build()
    #     filename = f"frame_{frame}.png"
    #     plt.savefig(filename)
    #     frames.append(filename)
    #     plt.close(fig)

    # # Create a GIF from the saved frames
    # with Image.open(frames[0]) as img:
    #     img.save("animation.gif", save_all=True, append_images=[Image.open(f) for f in frames[1:]], duration=100, loop=0)

    # # Remove the individual frame images
    # for frame in frames:
    #     os.remove(frame)





if __name__ == "__main__":
    main()
