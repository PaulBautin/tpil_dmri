#!/bin/bash
#
# run hippunfold
#
# Author: Paul Bautin
###################################################

# Uncomment for full verbose
set -x

# Immediately exit if error
set -e

# Exit if user presses CTRL+C (Linux) or CMD+C (OSX)
trap "echo Caught Keyboard Interrupt within script. Exiting now.; exit" INT

my_singularity_img='/home/pabaua/dev_scil/extractor_flow/containers/singularity_extractorflow.sif' # or .sif
my_main_nf='/home/pabaua/dev_scil/extractor_flow/main.nf'
my_input='/home/pabaua/dev_tpil/results/results_connectivity_prep/23-08-15_connectivity_prep/results'

NXF_DEFAULT_DSL=1 NXF_VER=21.10.6 nextflow run $my_main_nf --input $my_input \
    -with-singularity $my_singularity_img -resume --quick_registration true


