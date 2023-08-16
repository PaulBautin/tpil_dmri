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

sudo docker run -it --rm -v /home/pabaua/dev_tpil/data/22-08-19_dMRI_control_BIDS:/bids \
-v /home/pabaua/dev_tpil/results/hippunfold:/output khanlab/hippunfold:v1.2.0 /bids /output participant \
--participant_label pl024 --modality T1w --cores 4


NXF_DEFAULT_DSL=1 NXF_VER=21.10.6 nextflow run /home/pabaua/dev_scil/extractor_flow/main.nf --input /home/pabaua/dev_tpil/results/results_connectivity_prep/test_container/results -with-singularity /home/pabaua/dev_scil/extractor_flow/containers/singularity_extractorflow.sif -resume --quick_registration true


