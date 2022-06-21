#!/bin/bash

# This would run dmriqc_flow with the following parameters:
#   - profile input_qc


my_singularity_img='/home/pabaua/scil_dev/containers/scilus_1_3_0.img' # or .sif
my_main_nf='/home/pabaua/scil_dev/dmriqc_flow/main.nf'
my_input='/home/pabaua/tpil_dev/data/Data_dMRI_lowercase_CON'


NXF_VER=21.10.6 nextflow run $my_main_nf -profile input_qc --input $my_input \
    -with-singularity $my_singularity_img -resume
