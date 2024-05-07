#!/bin/bash

# This would run tractoflow with the following parameters:
#   - Dti_shells 0 and 300, Fodf_shells 0 and 2000.
#   - WM seeding


my_singularity_img='/home/pabaua/dev_scil/containers/containers_scilus_1.6.0.sif' # or .sif
my_main_nf='/home/pabaua/dev_scil/tractometry_flow/main.nf'
my_input='/home/pabaua/dev_tpil/data/tractometry_data/'


NXF_VER=21.10.6 nextflow run $my_main_nf --input $my_input \
    -with-singularity $my_singularity_img -resume --skip_projection_endpoints_metrics --use_provided_centroids -profile cbrain

