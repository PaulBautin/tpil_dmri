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
    p.add_argument('in_atlas',
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

    assert_inputs_exist(parser, args.in_atlas)
    assert_output_dirs_exist_and_empty(parser, args, args.out_dir,
                                       create_dir=True)

    atlas = nib.load(args.in_atlas)
    data_atlas = atlas.get_fdata()

    # Create mask mPFC
    mask_mPFC_NAC = (data_atlas == 27) | (data_atlas == 45) | (data_atlas == 47) | (data_atlas == 219) | (data_atlas == 223)
    atlas_mPFC_NAC = mask_mPFC_NAC * data_atlas
    labels = nib.Nifti1Image(atlas_mPFC_NAC, atlas.affine)
    nib.save(labels, '/home/pabaua/dev_tpil/data/connectflow_data/sub-pl010_ses-v1/sub-pl010_ses-v1__labels.nii.gz')



if __name__ == "__main__":
    main()