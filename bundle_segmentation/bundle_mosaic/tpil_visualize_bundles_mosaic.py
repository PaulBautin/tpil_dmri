#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from scripts import scil_visualize_bundles

"""
Visualize bundles from a list. The script will output a mosaic (image) with
screenshots, 6 views per bundle in the list.
"""
def main():
    paths_ref = [os.path.abspath(i) + '/' + i + '__ref_image.nii.gz' for i in
                 os.listdir('/home/pabaua/Desktop/test_bundle_mosaic') if 'sub' in i]
    paths_tracto = [os.path.abspath(i) + '/' + i + '__tractogram27.trk' for i in
                 os.listdir('/home/pabaua/Desktop/test_bundle_mosaic') if 'sub' in i]
    scil_visualize_bundles_mosaic.py paths_ref paths_tracto /home/pabaua/Desktop/test_bundle_mosaic/results/sub-pl010_ses-v1

if __name__ == '__main__':
    main()
