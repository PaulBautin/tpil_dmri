#!/bin/bash

# This would run dmriqc_flow with the following parameters:
#   - Dti_shells 0 and 300, Fodf_shells 0 and 2000.
#   - WM seeding


my_singularity_img='/home/pabaua/dev_scil/containers/scilus_1_3_0.img' # or .sif
my_main_nf='/home/pabaua/dev_scil/dmriqc_flow/main.nf'
my_input='/home/pabaua/dev_tpil/results/results_new_bundle/22-11-15_rbx/results_bundle'


NXF_VER=21.10.6 nextflow run $my_main_nf -profile rbx_qc --input $my_input \
    -with-singularity $my_singularity_img -resume
