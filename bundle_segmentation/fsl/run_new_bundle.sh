#!/bin/bash

# This would run the TPIL Bundle Segmentation Pipeline with the following resources:
#     Prebuild Singularity images: https://scil.usherbrooke.ca/pages/containers/
#     Brainnetome atlas in MNI space: https://atlas.brainnetome.org/download.html
#     FA template in MNI space: https://brain.labsolver.org/hcp_template.html


my_singularity_img='/home/pabaua/dev_scil/containers/containers_scilus_1.4.2.sif' # or .sif
my_main_nf='/home/pabaua/dev_tpil/tpil_dmri/bundle_segmentation/fsl/main.nf'
my_input='/home/pabaua/dev_tpil/results/results_tracto'
my_input_fs='/home/pabaua/dev_tpil/data/Freesurfer/22-09-21_t1_clbp_freesurfer_output'
my_template='/home/pabaua/dev_scil/atlas/mni_masked.nii.gz'

export SINGULARITY_BIND="/home/pabaua/dev_tpil/data/Freesurfer/license.txt:/opt/freesurfer-6.0.0/license.txt"
export SUBJECTS_DIR="/home/pabaua/dev_tpil/data/Freesurfer/22-09-21_t1_clbp_freesurfer_output"

nextflow run  \
  -with-singularity $my_singularity_img \
  $my_main_nf \
  --input $my_input \
  --input_fs $my_input_fs \
  --template $my_template \
  -resume


