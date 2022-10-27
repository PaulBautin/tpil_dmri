#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute bundle volume in mm3. This script supports anisotropic voxels
resolution. Volume is estimated by counting the number of voxels occupied by
the bundle and multiplying it by the volume of a single voxel.

This estimation is typically performed at resolution around 1mm3.
"""

from dipy.io.utils import (create_nifti_header, get_reference_info,
                           is_header_compatible)

import argparse
import json
import os
import nibabel as nib

import numpy as np
import matplotlib.pyplot as plt

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_json_args,
                             add_reference_arg,
                             assert_inputs_exist)
from scilpy.tractanalysis.streamlines_metrics import compute_tract_counts_map

from dipy.io.stateful_tractogram import StatefulTractogram
from dipy.tracking.streamlinespeed import (length, set_number_of_points)
from dipy.io.streamline import save_tractogram


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_bundle',
                   help='Fiber bundle file.')
    add_reference_arg(p)
    add_json_args(p)

    return p


def recolor_nifti(sft, labels, label_values):
    unique_labels = np.unique(labels)
    # Keep data as int to get the underlying voxel
    bundle_data_int = sft.streamlines.get_data().astype(np.int16)
    #metric = nib.load('/home/pabaua/tpil_dev/results/results/DCL_PT_PL007_V1/DTI_Metrics/DCL_PT_PL007_V1__fa.nii.gz')
    metric = nib.load('/home/pabaua/tpil_dev/results/results_rbx/DCL_PT_PL007_V1/Register_Anat/DCL_PT_PL007_V1__outputWarped.nii.gz')
    print(metric.header)
    print(is_header_compatible(metric, sft))
    metric_data = metric.get_fdata(dtype=np.float64)
    mask = np.zeros(metric_data.shape)
    for i in unique_labels:
        print(i)
        if label_values[i-1] <= 0.05:
            print('gggg')
            label_indices = bundle_data_int[labels == i]
            mask[label_indices[:, 0], label_indices[:, 1], label_indices[:, 2]] = 100
        #else:
        #    mask[label_indices[:, 0], label_indices[:, 1], label_indices[:, 2]] = 0.1
    img = nib.Nifti1Image(mask, metric.affine)
    nib.save(img, "/home/pabaua/tpil_dev/results/test.nii.gz")


def recolor_trk(sft, labels, label_values):
    unique_labels = np.unique(labels)
    print(unique_labels)
    cmap = plt.get_cmap('plasma')
    for i in unique_labels:
        if label_values[i - 1] <= 0.05:
            label_indices = np.where(labels == i)[0]
            labels[label_indices] = 1
        else:
            label_indices = np.where(labels == i)[0]
            labels[label_indices] = 0
    sft.data_per_point['color']._data = cmap(labels / np.max(labels))[:, 0:3] * 255
    save_tractogram(sft, "/home/pabaua/tpil_dev/results/test.trk")

def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.in_bundle, optional=args.reference)

    sft = load_tractogram_with_reference(parser, args, args.in_bundle)
    sft.to_vox()
    sft.to_corner()

    labels = np.load('/home/pabaua/tpil_dev/results/results_tractometry/DCL_PT_PL007_V1/Bundle_Label_And_Distance_Maps/DCL_PT_PL007_V1__AF_L_labels.npz')
    labels = labels['arr_0']

    # new labels
    label_values = [
0.9619643059154419,
0.590271605827511,
0.653998053625769,
0.6608125754689191,
0.04487018004262467,
0.3627572831090363,
0.6782243611961648,
0.538465116412292,
0.8975689225813437,
0.49545252924045424,
0.9742345959425471,
0.7946642981011551,
0.6015478174129538,
0.9773667297750623,
0.5848787114874202,
0.7575277133452805,
0.6351269204826573,
0.9994025704369984,
0.9291835529916277,
0.21898419147453801,
]

    recolor_nifti(sft, labels, label_values)
    recolor_trk(sft, labels, label_values)

if __name__ == '__main__':
    main()
