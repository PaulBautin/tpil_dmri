#!/bin/bash

# This would run extractorflow with the following parameters:



my_singularity_img='/home/pabaua/dev_scil/extractor_flow/containers/singularity_extractorflow.sif' # or .sif
my_main_nf='/home/pabaua/dev_scil/extractor_flow/main.nf'
my_input='/home/pabaua/dev_tpil/results/results_connectivity_prep/test_container/results'


nextflow run $my_main_nf --input $my_input \
    -with-singularity $my_singularity_img -resume
