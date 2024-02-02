#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Connectome Visualization and Smoothing Pipeline

This script provides a pipeline for the following tasks:
1. Mapping high-resolution structural connectivities using the Connectome Spatial Smoothing (CSS) package.
2. Applying a smoothing kernel to the connectome.
3. Filtering the connectome based on a specific brain structure.
4. Saving the modified data into a new CIFTI file.
5. Visualizing the CIFTI data on a surface plot.

Usage:
Run this script in a terminal, supplying the required command-line arguments.
>>> python CCS_2.py <in_tractogram> <surf_lh> <surf_rh> <cifti_file> <out_file>
"""

import argparse
import numpy as np
import scipy.sparse as sparse
import nibabel as nib
import matplotlib.pyplot as plt
from Connectome_Spatial_Smoothing import CSS as css
from scilpy.io.utils import add_overwrite_arg, assert_inputs_exist
from surfplot import Plot

from scilpy.io.streamlines import (reconstruct_streamlines,
                                   reconstruct_streamlines_from_hdf5)
from dipy.io.stateful_tractogram import (Origin, Space,
                                         StatefulTractogram)
from dipy.io.streamline import save_tractogram, load_tractogram
from scilpy.io.streamlines import load_tractogram_with_reference

#from neuromaps import nulls
#from neuromaps import stats as stats_neuromaps
#from scipy import stats
#from neuromaps.images import dlabel_to_gifti
#from neuromaps.datasets import fetch_fslr


def _build_arg_parser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter, description=__doc__
    )

    p.add_argument("in_tractogram", help="Path of the input tractograms (.trk).")
    p.add_argument("surf_lh", help="Path to left hemisphere surface")
    p.add_argument("surf_rh", help="Path to right hemisphere surface")
    p.add_argument("cifti_file", help="Path to cifti file")
    p.add_argument("out_file", help="Output file path (.trk).")
    p.add_argument("--weight_file", help="weight file path (.txt).", default=None)
    p.add_argument("--dps_name", help="Name of data per streamline to load.", default=None)
    add_overwrite_arg(p)
    return p


def plot_surface(cifti_file):
    """
    Plot a surface representation of a CIFTI file.

    Parameters
    ----------
    cifti_file : str or nib.Cifti2Image
        The CIFTI file or image object to plot.
    """
    surfaces = fetch_fslr()
    plt.style.use('dark_background')
    print(surfaces)
    lh, rh = surfaces['inflated']
    p = Plot(lh, rh, views=['lateral','medial', 'ventral'], zoom=1.2)
    p.add_layer(cifti_file, cbar=True, cmap='autumn')
    fig = p.build()
    plt.show(block=True)


def load_streamline_dps(sft, dps_name):
    """
    load streamline data

    Parameters
    ----------
    sft : statefull tractogram
        The tractogram with dps data
    """
    new_data_per_streamline = [s for s in sft.data_per_streamline[dps_name]]
    return new_data_per_streamline


def save_cifti_file(streamline_incidence, cifti_file, cifti_output):
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
    new_data = streamline_incidence[np.newaxis, ...] * np.ones_like(original_data)
    
    # Create a new CIFTI image object with the updated data
    new_img = nib.Cifti2Image(new_data, header=original_cifti.header, nifti_header=original_cifti.nifti_header)
    
    # Save the new CIFTI image to disk
    new_img.to_filename(cifti_output)
    
    return new_img


def filter_connectome(conn, cifti_file, structure='CIFTI_STRUCTURE_BRAIN_STEM'):
    """
    Filter the connectome matrix based on a specific brain structure.

    Parameters
    ----------
    conn : scipy.sparse.csr_matrix
        The connectome matrix.
    cifti_file : str
        Path to the CIFTI file.
    structure : str, optional
        The brain structure to filter by. Default is 'CIFTI_STRUCTURE_BRAIN_STEM'.

    Returns
    -------
    filtered_conn : scipy.sparse.csr_matrix
        The filtered connectome matrix.
    """
    
    # Load CIFTI file and extract brain models
    img_cifti = nib.load(cifti_file)
    brain_models = list(img_cifti.header.get_index_map(1).brain_models)
    
    # Create a dictionary for storing the number of voxel indices for each brain structure
    brain_structure_to_indices = {
        bm.brain_structure: np.array(bm.voxel_indices_ijk).shape[0] for bm in brain_models[2:]
    }
    
    # Find the position of the target structure in the dictionary
    struct_pos = list(brain_structure_to_indices).index(structure) + 1 if structure in brain_structure_to_indices else None

    if struct_pos is None:
        raise KeyError(f"Structure {structure} not found in CIFTI file.")
    
    # Compute cumulative sum of indices
    cumulative_indices = np.append(0, list(brain_structure_to_indices.values()))
    
    # Compute the range for the target structure using slicing
    start_range = 64984 + sum(cumulative_indices[:struct_pos])
    end_range = start_range + cumulative_indices[struct_pos]
    
    # Filter the connectome matrix
    return conn[:, start_range:end_range]


def streamlines_from_hdf5(in_tractogram, output_sft):
    import h5py
    hdf5_file = h5py.File(in_tractogram, 'r')

    # Keep track of the order of connections/streamlines in relation to the
    # tractogram as well as the number of streamlines for each connection.
    bundle_groups_len = []
    hdf5_keys = list(hdf5_file.keys())
    streamlines = []
    for key in hdf5_keys:
        tmp_streamlines = reconstruct_streamlines_from_hdf5(hdf5_file, key)
        streamlines.extend(tmp_streamlines)
        bundle_groups_len.append(len(tmp_streamlines))

    offsets_list = np.cumsum([0]+bundle_groups_len)
    ref = '/home/pabaua/dev_tpil/results/results_tracto/23-09-01_tractoflow_bundling/sub-pl007_ses-v1/Register_T1/sub-pl007_ses-v1__t1_warped.nii.gz'
    sft = StatefulTractogram(streamlines, ref,Space.VOX, origin=Origin.TRACKVIS)
    save_tractogram(sft, output_sft)
    return sft


def main():
    """
    Main function to execute the pipeline for connectivity mapping and visualization.
    """
    # Build argument parser and parse command-line arguments and verify that all input files exist
    parser = _build_arg_parser()
    args = parser.parse_args()
    assert_inputs_exist(parser, [args.in_tractogram, args.surf_lh, args.surf_rh, args.surf_rh, args.cifti_file])

    # Load the input files.
    print("Loading file {}".format(args.in_tractogram))
    sft = load_tractogram_with_reference(parser, args, args.in_tractogram)

    if args.dps_name:
        args.weights = load_streamline_dps(sft, args.dps_name)
    else:
        args.weights = None

    # Map high-resolution structural connectivity
    #output_sft = '/home/pabaua/dev_hiball/css_test/sft.trk'
    #sft_map = streamlines_from_hdf5('/home/pabaua/Desktop/sub-pl007_ses-v1__decompose.h5', output_sft)
    #args.in_tractogram = output_sft
    connectome = css.map_high_resolution_structural_connectivity(args.in_tractogram, args.surf_lh, args.surf_rh, threshold=2, cifti_file=args.cifti_file, subcortex=True, weights=args.weights)
    #connectome = abs(connectome)

    # Compute and apply the smoothing kernel to the connectome
    smoothing_kernel = css.compute_smoothing_kernel(
        args.surf_lh, args.surf_rh, fwhm=2, epsilon=0.1, cifti_file=args.cifti_file, subcortex=True)
    connectome = css.smooth_high_resolution_connectome(connectome, smoothing_kernel)

    # Save and load smoothed connectome to disk
    # sparse.save_npz('/home/pabaua/dev_hiball/css_test/smoothed_high_resolution_connectome.npz', connectome)
    # print("Loading connectome")
    # connectome = sparse.load_npz('/home/pabaua/dev_hiball/css_test/smoothed_high_resolution_connectome.npz')
    # print("Loaded connectome")

    # Filter the connectome based on a specific structure
    connectome = filter_connectome(connectome, args.cifti_file, structure='CIFTI_STRUCTURE_ACCUMBENS_LEFT')
    # connectome.data[np.isnan(connectome.data)] = 0.0
    # connectome.data[connectome.data < 0.01] = 0.0


    # Compute streamline incidence, normalize and save to a new CIFTI file
    # streamline_incidence = np.log1p(np.array(connectome.sum(axis=1))[..., 0]) / np.max(np.log1p(np.array(connectome.sum(axis=1))[..., 0]))
    streamline_incidence = np.log1p(np.array(connectome.sum(axis=1))[..., 0])#[:64984]
    print(streamline_incidence.shape)
    cifti_si = save_cifti_file(streamline_incidence, args.cifti_file, args.out_file)
    # cifti_si = nib.load(args.out_file)
    # Plot surface using the new CIFTI data
    # plot_surface(cifti_file=cifti_si)
    

    # curvature = nib.load('/home/pabaua/dev_tpil/results/results_new_bundle/23-10-19_accumbofrontal/results/sub-pl007_ses-v1/Fs_ciftify/sub-pl007_ses-v1/MNINonLinear/fsaverage_LR32k/sub-pl007_ses-v1.curvature.32k_fs_LR_resampled.dscalar.nii').get_fdata()
    # print(np.min(np.abs(curvature)))
    # print(np.max(np.abs(curvature)))
    # sulc = nib.load('/home/pabaua/dev_tpil/results/results_new_bundle/23-10-19_accumbofrontal/results/sub-pl007_ses-v1/Fs_ciftify/sub-pl007_ses-v1/MNINonLinear/fsaverage_LR32k/sub-pl007_ses-v1.curvature.32k_fs_LR_resampled_abs.gii')
    # cifti_si = nib.load('/home/pabaua/dev_tpil/results/results_map_projection/test_css.dscalar.gii')
    # print(np.min(cifti_si.agg_data()))
    # print(np.max(cifti_si.agg_data()))
    # nulls_t = nulls.alexander_bloch(cifti_si, atlas='fsLR', density='32k', n_perm=500, seed=1234)
    # corr, p = stats_neuromaps.compare_images(cifti_si, sulc, nulls=nulls_t)
    # print(corr)
    # print(p)
if __name__ == "__main__":
    main()
