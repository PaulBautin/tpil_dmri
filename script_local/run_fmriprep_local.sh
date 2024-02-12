#!/bin/bash

# This would run tractoflow with the following parameters:
#   - Dti_shells 0 and 300, Fodf_shells 0 and 2000.
#   - WM seeding


my_singularity_img='/home/pabaua/dev_scil/containers/fmriprep-23.2.0.simg' # or .sif
my_input='/home/pabaua/dev_tpil/data/BIDS_dataset_longitudinale/dataset/'
my_output='/home/pabaua/dev_tpil/results/results_fmriprep/test'
my_work='/home/pabaua/dev_tpil/results/results_fmriprep/test/work/'

my_licence_fs='/home/pabaua/dev_tpil/data/Freesurfer/license.txt'
export SINGULARITYENV_FS_LICENSE=$my_licence_fs
#singularity exec --cleanenv fmriprep.simg env | grep FS_LICENSE

my_participants='007 010'

singularity run $my_singularity_img $my_input $my_output participant -w $my_work --participant-label $my_participants --output-spaces T1w --cifti-output 91k --bids-filter-file /home/pabaua/dev_tpil/tpil_dmri/script_local/fmriprep_bids_filter_v1.json --fs-subjects-dir /home/pabaua/dev_tpil/data/Freesurfer/freesurfer_v1

#singularity run $my_singularity_img $my_input $my_output participant -w $my_work --output-spaces T1w --participant-label 012 --cifti-output 91k

#singularity run $my_singularity_img $my_input $my_output participant -w $my_work --output-spaces T1w --participant-label 002 --cifti-output 91k --me-t2s-fit-method curvefit --me-output-echos
