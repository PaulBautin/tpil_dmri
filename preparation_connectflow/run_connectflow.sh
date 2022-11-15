#!/bin/bash



my_singularity_img='/home/pabaua/dev_scil/containers/scilus_1_3_0.img' # or .img
my_main_nf='/home/pabaua/dev_scil/connectoflow/main.nf'
my_input='/home/pabaua/dev_tpil/data/connectflow_data'
label_list='/home/pabaua/dev_tpil/data/connectflow_data/atlas_brainnetome_v4_labels_list.txt'
template='/home/pabaua/Downloads/fsl-6.0.5.2-sources/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz'


NXF_DEFAULT_DSL=1 nextflow run $my_main_nf --input $my_input --labels_list $label_list --template $template \
    -with-singularity $my_singularity_img -resume

