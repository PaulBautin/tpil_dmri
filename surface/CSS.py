#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to load and transform a FSL FIRST surface (VTK supported),
This script is using ANTs transform (affine.txt, warp.nii.gz).

Best usage with ANTs from T1 to b0:
> ConvertTransformFile 3 output0GenericAffine.mat vtk_transfo.txt --hm
> scil_transform_surface.py L_Accu.vtk L_Accu_diff.vtk\\
    --ants_affine vtk_transfo.txt --ants_warp warp.nii.gz

The input surface needs to be in *T1 world LPS* coordinates
(aligned over the T1 in MI-Brain).
The resulting surface should be aligned *b0 world LPS* coordinates
(aligned over the b0 in MI-Brain).
"""

import itertools
import os
import sys
import numpy as np
import scipy.sparse as sparse
from nilearn import plotting
import nibabel as nib
import matplotlib.pyplot as plt

# Additionally, import the CSS package
from Connectome_Spatial_Smoothing import CSS as css
#import hcp_utils as hcp
import scipy
from dipy.io.streamline import load_tractogram


def get_xyz_hem_surface(hem_surface_file, brain_model_index, cifti_file):
    """
    returns the xyz mm coordinates of all brainordinates in that hemisphere's surface mesh (excludes medial wall)
    """
    img = nib.load(cifti_file)

    brain_models = [x for x in img.header.get_index_map(1).brain_models]

    hem_surface = nib.load(hem_surface_file)

    return hem_surface.darrays[0].data[brain_models[brain_model_index].vertex_indices]

def get_xyz_surface(left_surface_file, right_surface_file, cifti_file):
    """
    returns the xyz mm coordinates of all brainordinates in the surface mesh (excludes medial wall)
    """
    # left cortex
    leftxyz = get_xyz_hem_surface(left_surface_file, 0, cifti_file=cifti_file)

    # right cortex
    rightxyz = get_xyz_hem_surface(right_surface_file, 1, cifti_file=cifti_file)

    return np.vstack([leftxyz, rightxyz])



def get_xyz(left_surface_file, right_surface_file, cifti_file, subject=None):
    """
    returns the xyz mm coordinates of all brainordinates in the cifti file (excludes medial wall, but includes subcortex)
    """
    img = nib.load(cifti_file)

    brain_models = [x for x in img.header.get_index_map(1).brain_models]

    # left cortex
    leftxyz = get_xyz_hem_surface(left_surface_file, 0, cifti_file=cifti_file)

    # right cortex
    rightxyz = get_xyz_hem_surface(right_surface_file, 1, cifti_file=cifti_file)
    print(img.header.get_index_map(1).volume.transformation_matrix_voxel_indices_ijk_to_xyz.matrix)

    # subcortical regions
    #subijk = np.array(list(itertools.chain.from_iterable([(x.voxel_indices_ijk) for x in brain_models[2:]])))
    #subxyz = nib.affines.apply_affine(img.header.get_index_map(1).volume.transformation_matrix_voxel_indices_ijk_to_xyz.matrix, subijk)
    img = nib.load("/home/pabaua/dev_tpil/results/results_fslfirst/clbp/"+ subject+ "/Subcortex_segmentation/"+ subject+ "_all_fast_firstseg.nii.gz")
    subijk = np.argwhere(img.get_fdata() != 0)
    subxyz = nib.affines.apply_affine(img.affine, subijk)
    

    xyz = np.vstack([leftxyz, rightxyz, subxyz])

    return xyz




def get_endpoint_distances_from_tractography(track_file,
                                              left_surface_file,
                                              right_surface_file,
                                              cifti_file,
                                              warp_file=None,
                                              subcortex=False,
                                              subject=None
                                              ):
    """
    Returns the streamline endpoint distances from closest vertex on cortical surface mesh
    and the closest vertex index. Additionally warps the endpoints before aligning to the
    surface mesh, if a warp file is provided. This is useful when mapping a native
    tractography file to a standard space surface mesh.
    """
    # load the track file streamlines
    print('loading tractography file.')
    #tracks = nib.streamlines.load(track_file)
    tracks = load_tractogram(track_file, reference="/home/pabaua/dev_tpil/results/results_tracto/23-09-01_tractoflow_bundling/"+ subject+ "/Crop_T1/"+ subject+ "__t1_bet_cropped.nii.gz")
    print('track file loaded: {}'.format(track_file))
    tracks.to_center()
    

    # extract streamline endpoints
    starts = np.array([stream[0] for stream in tracks.streamlines])
    ends = np.array([stream[-1] for stream in tracks.streamlines])
    print('endpoints extracted: #{}'.format(len(starts)))

    # extract cortical surface coordinates
    if subcortex:
        xyz = get_xyz(left_surface_file, right_surface_file, cifti_file=cifti_file, subject=subject)
    else:
        xyz = get_xyz_surface(left_surface_file, right_surface_file, cifti_file=cifti_file)
    #xyz = np.vstack([nib.load(left_surface_file).agg_data('pointset'), nib.load(right_surface_file).agg_data('pointset')])
    plt.scatter(xyz[:,0], xyz[:,1])
    plt.scatter(starts[:,0], starts[:,1])
    plt.axis('equal')
    plt.show()

    # store the coordinates in a kd-tree data structure to locate closest point faster
    kdtree = scipy.spatial.cKDTree(xyz)

    # locate closest surface points to every endpoint
    start_dists, start_indices = kdtree.query(starts)
    end_dists, end_indices = kdtree.query(ends)
    print('closest brainordinates located')

    return (start_dists, start_indices, end_dists, end_indices, len(xyz))


def get_half_incidence_matrices_from_endpoint_distances(start_dists,
                                                         start_indices,
                                                         end_dists,
                                                         end_indices,
                                                         node_count,
                                                         threshold,
                                                         weights=None):
    """
    Returns two half incidence matrices in a sparse format (CSR) after
    filtering the streamlines that are far (>2mm) from their closest vertex.
    """

    if weights is None:
        weights = np.ones(len(start_dists))
    elif (type(weights) == str):
        # load text file contents (this could be sift weights files generated by mrtrix)
        weights = np.genfromtxt(weights)

    # mask points that are further than the threshold from all surface coordinates
    outlier_mask = (start_dists > threshold) | (end_dists > threshold)
    print('outliers located: #{} outliers ({}%, with threshold {}mm)'.format(
        sum(outlier_mask),
        (100 * sum(outlier_mask)) / len(outlier_mask),
        threshold,
    ))

    # create a sparse incidence matrix
    print('creating sparse incidence matrix')
    start_dict = {}
    end_dict = {}
    indices = (i for i in range(len(outlier_mask)) if not outlier_mask[i])
    for l, i in enumerate(indices):
        start_dict[(start_indices[i], l)] = start_dict.get((start_indices[i], l), 0) + weights[i]
        end_dict[(end_indices[i], l)] = end_dict.get((end_indices[i], l), 0) + 1

    start_inc_mat = sparse.dok_matrix(
        (
            node_count,
            (len(outlier_mask) - outlier_mask.sum())
        ),
        dtype=np.float32
    )

    for key in start_dict:
        start_inc_mat[key] = start_dict[key]

    end_inc_mat = sparse.dok_matrix(
        (
            node_count,
            (len(outlier_mask) - outlier_mask.sum())
        ),
        dtype=np.float32
    )
    for key in end_dict:
        end_inc_mat[key] = end_dict[key]

    print('sparse matrix generated')

    return (start_inc_mat.tocsr(), end_inc_mat.tocsr())

def get_adjacency_from_half_incidence_matrices(U, V, stat='sum'):
    """
    return a sparse adjacency matrix A from the two halfs of incidence matrix U & V.
    """
    A = U.dot(V.T)
    A = A + A.T
    if stat == 'mean':
        # compute average instead of sum
        A_div = (U != 0).astype(int).dot(((V != 0).astype(int)).T)
        A_div = A_div + A_div.T
        A.data = A.data / A_div.data
    return A

def plot_surface(surf_data_left, surf_data_right, cifti_file):
    from surfplot import Plot
    from neuromaps.datasets import fetch_fslr
    # Or use one of the surface files included in the neuromaps package 
    # Here we are using the 32k FsLR (a symmetric version 
    # fsaverage space template with ~32k verticies in each hemisphere
    surfaces = fetch_fslr()
    lh, rh = surfaces['inflated']

    # Generate plot
    # 'ventral', 'anterior'
    p = Plot(lh, rh, views=['lateral','medial', 'ventral', 'anterior'], zoom=1.2, flip=True)
    p.add_layer({'left': surf_data_left/(np.max([surf_data_left, surf_data_right])), 'right': surf_data_right/(np.max([surf_data_left, surf_data_right]))}, cbar=True, cmap='inferno')
    lh_BN = '/home/pabaua/dev_tpil/data/BN/BN_Atlas_freesurfer/fsaverage/fsaverage_LR32k/fsaverage.L.BN_Atlas.32k_fs_LR.label.gii'
    rh_BN = '/home/pabaua/dev_tpil/data/BN/BN_Atlas_freesurfer/fsaverage/fsaverage_LR32k/fsaverage.R.BN_Atlas.32k_fs_LR.label.gii'
    #p.add_layer({'left': lh_BN, 'right': rh_BN}, cbar=False, cmap='inferno', as_outline=True)
    #p.add_layer(cifti_file, cbar=True, cmap='inferno')
    fig = p.build()
    plt.show(block=True)

def save_cifti_file(streamline_incidence, cifti_file, subject):
    cifti_data = nib.load(cifti_file).get_fdata()
    cifti_data = streamline_incidence[np.newaxis, ...] * cifti_data
    #cifti_data = np.pad(streamline_incidence[np.newaxis, ...],((0, 0), (0, 91282-59412)), 'constant', constant_values=1) * cifti_data
    new_img = nib.Cifti2Image(cifti_data, header=nib.load(cifti_file).header, nifti_header=nib.load(cifti_file).nifti_header)
    new_img.to_filename("/home/pabaua/dev_tpil/"+subject+"stremlines.dscalar.nii")
    #new_img.to_filename("/home/pabaua/dev_tpil/stremlines_10.dscalar.nii")
    streamline_incidence = nib.load("/home/pabaua/dev_tpil/"+'sub-pl007_ses-v1'+"stremlines.dscalar.nii").get_fdata()[..., :59412] +nib.load("/home/pabaua/dev_tpil/"+'sub-pl010_ses-v1'+"stremlines.dscalar.nii").get_fdata()[..., :59412] + nib.load("/home/pabaua/dev_tpil/"+subject+"stremlines.dscalar.nii").get_fdata()[..., :59412]
    streamline_incidence = streamline_incidence.T[:,0]
    #return streamline_incidence

def main():
    # Get paths for all files:
    css_path = "/home/pabaua/dev_hiball/connectome-spatial-smoothing"
    main_dir = os.path.join(os.path.abspath(css_path), 'code/Connectome_Spatial_Smoothing')

    subject = 'sub-pl007_ses-v1'
    #tractography_file_nac = os.path.abspath('/home/pabaua/dev_hiball/css_test/test_t1/' + subject+ '__NAc_proj.trk')
    tractography_file_nac = os.path.abspath('/home/pabaua/dev_tpil/results/results_new_bundle/23-10-19_accumbofrontal/results/' + subject+ '/Tractography_registration/' + subject+ '__pft_tracking_prob_wm_seed_0_t1.trk')
    left_native_surface_file = os.path.abspath('/home/pabaua/dev_tpil/data/Freesurfer/22-09-21_t1_clbp_hcp_output/' + subject+ '/T1w/fsaverage_LR32k/' + subject + '.L.white.32k_fs_LR.surf.gii')
    right_native_surface_file = os.path.abspath('/home/pabaua/dev_tpil/data/Freesurfer/22-09-21_t1_clbp_hcp_output/' + subject+ '/T1w/fsaverage_LR32k/' + subject+ '.R.white.32k_fs_LR.surf.gii')
    #cifti_file = os.path.abspath('{}/data/templates/cifti/ones.dscalar.nii'.format(main_dir))
    cifti_file = os.path.abspath('/home/pabaua/dev_hiball/css_test/test_t1/' + subject+ '.dscalar.nii')
    img_cifti = nib.load(cifti_file)
    brain_models = [x for x in img_cifti.header.get_index_map(1).brain_models]
    ijk = {x.brain_structure: np.array(x.voxel_indices_ijk).shape[0] for x in brain_models[2:]}
    # ['CIFTI_STRUCTURE_ACCUMBENS_LEFT', 'CIFTI_STRUCTURE_ACCUMBENS_RIGHT', 'CIFTI_STRUCTURE_AMYGDALA_LEFT', 'CIFTI_STRUCTURE_AMYGDALA_RIGHT', 'CIFTI_STRUCTURE_BRAIN_STEM', 'CIFTI_STRUCTURE_CAUDATE_LEFT', 'CIFTI_STRUCTURE_CAUDATE_RIGHT', 'CIFTI_STRUCTURE_HIPPOCAMPUS_LEFT', 'CIFTI_STRUCTURE_HIPPOCAMPUS_RIGHT', 'CIFTI_STRUCTURE_PALLIDUM_LEFT', 'CIFTI_STRUCTURE_PALLIDUM_RIGHT', 'CIFTI_STRUCTURE_PUTAMEN_LEFT', 'CIFTI_STRUCTURE_PUTAMEN_RIGHT', 'CIFTI_STRUCTURE_THALAMUS_LEFT', 'CIFTI_STRUCTURE_THALAMUS_RIGHT']
    struct = np.argwhere([x == 'CIFTI_STRUCTURE_BRAIN_STEM' for x in ijk.keys()])[0][0] + 1
    list_struct = np.append(0,list(ijk.values()))
    print(list_struct[:struct])
    print(list_struct[struct-1])
    range_struct = range(64984 + sum(list_struct[:struct]), 64984 + sum(list_struct[:struct]) + list_struct[struct-1])
    print(range_struct)
    #print(img_cifti.get_fdata()[0, ...][ijk].shape)
    #print([x.volume for x in brain_models])

    
    # Get paths for all files:
    #main_dir = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())), 'code/Connectome_Spatial_Smoothing')
    #brain_atlas_file = os.path.abspath('/home/pabaua/dev_hiball/connectome-spatial-smoothing/code/Connectome_Spatial_Smoothing/data/templates/atlas/Glasser360.32k_fs_LR.dlabel.nii')
    #brain_atlas_file = os.path.abspath('/home/pabaua/dev_tpil/data/BN/BN_Atlas_freesurfer/fsaverage/fsaverage_LR32k/fsaverage.BN_Atlas.32k_fs_LR.dlabel.nii')

    # Map high-resolution connectome onto native surfaces:
    #connectome_nac = css.map_high_resolution_structural_connectivity(tractography_file_nac, left_native_surface_file, right_native_surface_file, threshold=2, cifti_file=cifti_file, subcortex=True)
    #high_resolution_connectome_nac = css.map_high_resolution_structural_connectivity(tractography_file_nac, left_MNI_surface_file, right_MNI_surface_file, threshold=5)
    #(native_high_resolution_connectome)
    #sparse.save_npz('/home/pabaua/dev_hiball/css_test/surface_density/native_high_resolution_connectome_nac_2.npz', native_high_resolution_connectome_nac)
    #sparse_matrix = sparse.load_npz('/home/pabaua/dev_hiball/css_test/surface_density/native_high_resolution_connectome_nac_2.npz')
    #smoothing_kernel = css.compute_smoothing_kernel(left_native_surface_file, right_native_surface_file, fwhm=3, epsilon=0.1, cifti_file=cifti_file, subcortex=True)
    #connectome_nac = css.smooth_high_resolution_connectome(connectome_nac, smoothing_kernel)
    
    # print(smoothed_high_resolution_connectome)
    #labels, parcellation_matrix = css.parcellation_characteristic_matrix(atlas_file=brain_atlas_file)
    #atlas_connectome = css.downsample_high_resolution_structural_connectivity_to_atlas(native_high_resolution_connectome_nac, parcellation_matrix)
    #streamline_incidence_atlas = np.array(atlas_connectome.sum(axis=1))[..., 0]
    #print(nib.load(brain_atlas_file).get_fdata()[0, ...].shape)

    # res = get_endpoint_distances_from_tractography(tractography_file_nac, left_native_surface_file, right_native_surface_file,cifti_file=cifti_file, subject=subject, subcortex=True)
    # res = get_half_incidence_matrices_from_endpoint_distances(*res,threshold=2)
    # res = get_adjacency_from_half_incidence_matrices(*res)
    # print(res.shape)
    # smoothing_kernel = css.compute_smoothing_kernel(left_native_surface_file, right_native_surface_file, fwhm=5, epsilon=0.1, subcortex=True)
    # print(smoothing_kernel.shape)
    # smoothed_high_resolution_connectome = css.smooth_high_resolution_connectome(res, smoothing_kernel)
    # streamline_incidence = np.array(res.sum(axis=1))[..., 0]
    # print(streamline_incidence.shape)

    # save cifti file
    #streamline_incidence = save_cifti_file(streamline_incidence, cifti_file, subject)
    #sparse.save_npz('/home/pabaua/dev_hiball/css_test/surface_density/smoothed_high_resolution_connectome.npz', connectome_nac)
    connectome_nac = sparse.load_npz('/home/pabaua/dev_hiball/css_test/surface_density/smoothed_high_resolution_connectome.npz')
    import bct.algorithms as bct_alg
    import bct.utils as bct
    connectome_nac = bct_alg.distance_wei(connectome_nac)
    connectome_nac = connectome_nac[:,range_struct]
    streamline_incidence = np.log1p(np.array(connectome_nac.sum(axis=1))[..., 0])
    print(streamline_incidence.shape) 
    #surf_data_left = hcp.left_cortex_data(streamline_incidence, fill=0)
    #surf_data_right = hcp.right_cortex_data(streamline_incidence, fill=0)
    surf_data_left = streamline_incidence[:32492]
    surf_data_right = streamline_incidence[32492:(32492*2)]
    print(surf_data_left.shape)
    plot_surface(surf_data_left, surf_data_right, cifti_file=cifti_file)

    

if __name__ == "__main__":
    main()