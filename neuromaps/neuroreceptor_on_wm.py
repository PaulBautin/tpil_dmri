#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from neuromaps import images, resampling, plotting
import nibabel as nib
import numpy as np
import glob
import os
from os.path import dirname as up
import pandas as pd
from neuromaps import datasets
from neuromaps import images, nulls
from neuromaps import stats

import scilpy


def main():
    """
    Main function to execute the pipeline for connectivity mapping and visualization.
    """
    jaworska = datasets.fetch_annotation(source='jaworska2020')
    jaworska_img = images.load_nifti(jaworska)
    print(jaworska_img)


    kaller = datasets.fetch_annotation(source='kaller2017')
    thickness_r, kaller_r = resampling.resample_images(src=["/home/pabaua/dev_hiball/avg_surf_lh.gii","/home/pabaua/dev_hiball/avg_surf_rh.gii"], trg=kaller,
                                                src_space='fsaverage',
                                                trg_space='MNI152',
                                                resampling='transform_to_alt',
                                                alt_spec=('fsaverage', '10k'))

    kantonen = datasets.fetch_annotation(source='kantonen2020')
    thickness_r, kantonen_r = resampling.resample_images(src=["/home/pabaua/dev_hiball/avg_surf_lh.gii","/home/pabaua/dev_hiball/avg_surf_rh.gii"], trg=kantonen,
                                                src_space='fsaverage',
                                                trg_space='MNI152',
                                                resampling='transform_to_alt',
                                                alt_spec=('fsaverage', '10k'))

if __name__ == "__main__":
    main()