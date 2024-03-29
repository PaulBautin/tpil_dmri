#!/bin/bash

# This would run tractoflow with the following parameters:
#   - Dti_shells 0 and 300, Fodf_shells 0 and 2000.
#   - WM seeding


my_singularity_img='/home/pabaua/dev_scil/containers/containers_scilus_1.6.0.sif' # or .sif
my_main_nf='/home/pabaua/dev_scil/tractoflow/main.nf'
my_input='/home/pabaua/dev_tpil/data/BIDS_dataset_longitudinale/dataset/'


NXF_VER=21.10.6 nextflow run $my_main_nf --bids $my_input \
    -with-singularity $my_singularity_img -resume -with-report report.html \
    --dti_shells "0 1000" --fodf_shells "0 1000 2000" -profile bundling \
    --bidsignore '/home/pabaua/dev_tpil/data/BIDS_dataset_longitudinale/dataset/.bidsignore'

