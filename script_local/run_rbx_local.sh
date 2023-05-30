#!/bin/bash

# This would run tractoflow with the following parameters:
#   - Dti_shells 0 and 300, Fodf_shells 0 and 2000.
#   - WM seeding


my_singularity_img='/home/pabaua/dev_scil/containers/containers_scilus_1.5.0.sif' # or .sif
my_main_nf='/home/pabaua/dev_scil/rbx_flow/main.nf'
my_input='/home/pabaua/dev_tpil/data/Pascal'
my_atlas_config='/home/pabaua/dev_scil/atlas/config/config_ind.json'
my_atlas_anat='/home/pabaua/dev_scil/atlas/atlas_2023/mni_masked.nii.gz'
my_atlas_dir='/home/pabaua/dev_scil/atlas/atlas_2023/atlas'
my_atlas_centroids='/home/pabaua/dev_scil/atlas/atlas_2023/centroids'



NXF_VER=21.10.6 NXF_DEFAULT_DSL=1 nextflow run $my_main_nf --input $my_input \
    -with-singularity $my_singularity_img -resume -with-report report.html \
    --atlas_config $my_atlas_config --atlas_anat $my_atlas_anat --atlas_directory $my_atlas_dir --atlas_centroids $my_atlas_centroids
