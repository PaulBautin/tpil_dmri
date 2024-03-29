#!/bin/bash

# This would run the TPIL Bundle Segmentation Pipeline with the following resources:
#     Prebuild Singularity images: https://scil.usherbrooke.ca/pages/containers/
#     Brainnetome atlas in MNI space: https://atlas.brainnetome.org/download.html
#     FA template in MNI space: https://brain.labsolver.org/hcp_template.html


my_singularity_img='/home/pabaua/dev_scil/containers/scilus_1_3_0.img' # or .sif
my_main_nf='/home/pabaua/dev_tpil/tpil_dmri/bundle_segmentation/original/main.nf'
my_input='/home/pabaua/dev_tpil/data/data_new_bundle'
my_atlas='/home/pabaua/dev_tpil/data/BN/BN_Atlas_for_FSL/Brainnetome/BNA-maxprob-thr0-1mm.nii.gz'
my_template='/home/pabaua/dev_tpil/data/HCP/FSL_HCP1065_FA_1mm.nii.gz'


nextflow run $my_main_nf --input $my_input --atlas $my_atlas \
    -with-singularity $my_singularity_img -resume  --template $my_template --processes 2


my_main_nf_qc='/home/pabaua/dev_scil/dmriqc_flow/main.nf'
my_input_qc='/home/pabaua/dev_tpil/results/results_new_bundle/test/results_bundle'

NXF_VER=21.10.6 nextflow run $my_main_nf_qc -profile rbx_qc --input $my_input_qc \
    -with-singularity $my_singularity_img -resume

