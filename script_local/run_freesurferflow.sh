#!/bin/bash



my_singularity_img='/home/pabaua/dev_scil/freesurfer_flow/scil_freesurfer.img' # or .img
my_main_nf='/home/pabaua/dev_scil/freesurfer_flow/main.nf'
my_input='/home/pabaua/dev_tpil/data/clbp_t1_freesurfer_input/sub-pl007_ses-v1_regi'
atlas_utils='/home/pabaua/dev_scil/freesurfer_flow/FS_BN_GL_SF_utils'


NXF_DEFAULT_DSL=1 nextflow run $my_main_nf --root_fs_output $my_input --atlas_utils_folder $atlas_utils \
    -with-singularity $my_singularity_img -resume -with-report report.html \
    --compute_lausanne_multiscale false

