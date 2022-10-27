#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute a density map for each connection from a hdf5 file.
Typically use after scil_decompose_connectivity.py in order to obtain the
average density map of each connection to allow the use of --similarity
in scil_compute_connectivity.py.

This script is parallelized, but will run much slower on non-SSD if too many
processes are used. The output is a directory containing the thousands of
connections:
out_dir/
    |-- LABEL1_LABEL1.nii.gz
    |-- LABEL1_LABEL2.nii.gz
    |-- [...]
    |-- LABEL90_LABEL90.nii.gz
"""

import argparse
import itertools
import multiprocessing
import os

from dipy.io.streamline import save_tractogram

import h5py
import numpy as np
import nibabel as nib

from dipy.io.stateful_tractogram import Space, Origin, StatefulTractogram

from scilpy.io.streamlines import reconstruct_streamlines_from_hdf5
from scilpy.io.utils import (add_overwrite_arg,
                             add_processes_arg,
                             assert_inputs_exist,
                             assert_output_dirs_exist_and_empty,
                             validate_nbr_processes)
from dipy.io.utils import create_nifti_header, get_reference_info



def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_hdf5', nargs='+',
                   help='List of HDF5 filenames (.h5) from '
                        'scil_decompose_connectivity.py.')
    p.add_argument('out_dir',
                   help='Path of the output directory.')

    p.add_argument('--binary', action='store_true',
                   help='Binarize density maps before the population average.')

    add_processes_arg(p)
    add_overwrite_arg(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.in_hdf5)
    assert_output_dirs_exist_and_empty(parser, args, args.out_dir,
                                       create_dir=True)

    key_conn = '210_234'
    in_hdf5_file = h5py.File(args.in_hdf5[0], 'r')
    affine = in_hdf5_file.attrs['affine']
    dimensions = in_hdf5_file.attrs['dimensions']
    voxel_sizes = in_hdf5_file.attrs['voxel_sizes']
    streamlines = reconstruct_streamlines_from_hdf5(in_hdf5_file, key_conn)


    header = create_nifti_header(affine, dimensions, voxel_sizes)
    sft = StatefulTractogram(streamlines, header, Space.VOX, origin=Origin.TRACKVIS)
    save_tractogram(sft, args.out_dir)



if __name__ == "__main__":
    main()