#!/bin/bash

# This would run the TPIL Bundle Segmentation Pipeline with the following resources:
#     Prebuild Singularity images: https://scil.usherbrooke.ca/pages/containers/
#     Brainnetome atlas in MNI space: https://atlas.brainnetome.org/download.html
#     FA template in MNI space: https://brain.labsolver.org/hcp_template.html


my_singularity_img='/home/pabaua/dev_scil/containers/containers_scilus_1.4.2.sif' # or .sif
my_main_nf='/home/pabaua/dev_tpil/tpil_dmri/bundle_segmentation/fsl/main.nf'
my_input='/home/pabaua/dev_tpil/results/results_tracto'



nextflow run $my_main_nf --input $my_input -with-singularity $my_singularity_img -resume


