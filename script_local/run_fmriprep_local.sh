#!/bin/bash

# This would run tractoflow with the following parameters:
#   - Dti_shells 0 and 300, Fodf_shells 0 and 2000.
#   - WM seeding


my_singularity_img='/home/pabaua/dev_scil/containers/fmriprep-23.2.0a2.simg' # or .sif
my_main_nf='/home/pabaua/dev_scil/tractoflow/main.nf'
my_input='/home/pabaua/dev_tpil/data/BIDS_dataset/dataset'
my_output='/home/pabaua/dev_tpil/results/results_fmriprep/23-11-26_fmriprep/'
my_work='/home/pabaua/dev_tpil/results/results_fmriprep/23-11-26_fmriprep/work/'

my_licence_fs='/home/pabaua/dev_tpil/data/Freesurfer/license.txt'
export SINGULARITYENV_FS_LICENSE=$my_licence_fs
#singularity exec --cleanenv fmriprep.simg env | grep FS_LICENSE


#singularity run $my_singularity_img $my_input $my_output participant -w $my_work --output-spaces T1w --participant-label 002 --cifti-output 91k --project-goodvoxels

singularity run $my_singularity_img $my_input $my_output participant -w $my_work --output-spaces T1w --participant-label 007 --fs-no-reconall
