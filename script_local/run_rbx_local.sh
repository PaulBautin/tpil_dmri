#!/bin/bash

# This would run tractoflow with the following parameters:
#   - Dti_shells 0 and 300, Fodf_shells 0 and 2000.
#   - WM seeding


my_singularity_img='/home/pabaua/scil_dev/containers/scilus_1_3_0.img' # or .sif
my_main_nf='/home/pabaua/scil_dev/rbx_flow/main.nf'
my_input='/home/pabaua/tpil_dev/data/dataset_sansan_rbx'
my_atlas_config='/home/pabaua/scil_dev/atlas/config.json'
my_atlas_anat='/home/pabaua/scil_dev/atlas/mni_masked.nii.gz'
my_atlas_dir='/home/pabaua/scil_dev/atlas/atlas'
my_atlas_centroids='/home/pabaua/scil_dev/atlas/centroids'



NXF_DEFAULT_DSL=1 nextflow run $my_main_nf --input $my_input \
    -with-singularity $my_singularity_img -resume -with-report report.html \
    --atlas_config $my_atlas_config --atlas_anat $my_atlas_anat --atlas_directory $my_atlas_dir --atlas_centroids $my_atlas_centroids
