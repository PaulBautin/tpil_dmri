#!/bin/bash

# This would run tractoflow with the following parameters:
#   - Dti_shells 0 and 300, Fodf_shells 0 and 2000.
#   - WM seeding


my_singularity_img='/home/pabaua/scil_dev/containers/scilus_1_3_0.img' # or .sif
my_main_nf='/home/pabaua/tpil_dev/github/tpil_dmri/bundle_for_tractometry/main.nf'
my_input='/home/pabaua/tpil_dev/results/data_new_bundle'
my_atlas='/home/pabaua/scil_dev/atlas/BN_Atlas_246_1mm.nii.gz'


nextflow run $my_main_nf --input $my_input --atlas $my_atlas \
    -with-singularity $my_singularity_img -resume

