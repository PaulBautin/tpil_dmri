#!/bin/bash

# This would run the TPIL Bundle Segmentation Pipeline with the following resources:
#     Prebuild Singularity images: https://scil.usherbrooke.ca/pages/containers/
#     Brainnetome atlas in MNI space: https://atlas.brainnetome.org/download.html
#     FA template in MNI space: https://brain.labsolver.org/hcp_template.html


my_singularity_img='/home/pabaua/dev_scil/containers/scilus_1_3_0.img' # or .sif
my_main_nf='/home/pabaua/dev_tpil/github/tpil_dmri/bundle_for_tractometry/main.nf'
my_input='/home/pabaua/dev_tpil/data/data_new_bundle'
my_atlas='/home/pabaua/dev_tpil/data/data_new_bundle/BNA-maxprob-thr0-1mm.nii.gz'
my_template='/home/pabaua/dev_tpil/data/data_new_bundle/FSL_HCP1065_FA_1mm.nii.gz'


nextflow run $my_main_nf --input $my_input --atlas $my_atlas \
    -with-singularity $my_singularity_img -resume  --template $my_template
