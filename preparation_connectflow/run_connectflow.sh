#!/bin/bash



my_singularity_img='/home/pabaua/dev_scil/containers/containers_scilus_1.5.0.sif' # or .img
my_main_nf='/home/pabaua/dev_scil/connectoflow/main.nf'
my_input='/home/pabaua/dev_tpil/results/results_connectivity_prep/test_container/results'
label_list='/home/pabaua/dev_tpil/tpil_network_analysis/connectivity_prep/freesurfer_data/atlas_brainnetome_first_label_list.txt'
template='/home/pabaua/dev_tpil/tpil_network_analysis/connectivity_prep/freesurfer_data/mni_masked.nii.gz'


NXF_DEFAULT_DSL=1 NXF_VER=21.10.6 nextflow run $my_main_nf --input $my_input --labels_list $label_list --template $template \
    --apply_t1_labels_transfo false --nbr_subjects_for_avg_connections 20 -with-singularity $my_singularity_img -resume

