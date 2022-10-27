#!/bin/bash



my_singularity_img='/home/pabaua/dev_scil/containers/scilus_1_3_0.img' # or .img
my_main_nf='/home/pabaua/dev_scil/connectoflow/main.nf'
my_input='/home/pabaua/dev_tpil/data/connectflow_data'
label_list='/home/pabaua/dev_scil/containers/FS_BN_GL_SF_utils/freesurfer_utils/atlas_brainnetome_v4_labels_list.txt'
template='/home/pabaua/dev_scil/atlas/mni_masked.nii.gz'


NXF_DEFAULT_DSL=1 nextflow run $my_main_nf --input $my_input --label_list $label_list --template $template \
    -with-singularity $my_singularity_img -resume

