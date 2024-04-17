#!/bin/bash




singularity exec --no-home \
                 -B /home/pabaua/dev_tpil/data/BIDS_multiecho/dataset_multiecho:/data \
                 -B /home/pabaua/dev_tpil/results/results_fastsurfer:/output \
                 -B /home/pabaua/dev_tpil/data/Freesurfer:/fs \
                  ./fastsurfer-gpu.sif \
                  /fastsurfer/run_fastsurfer.sh \
                  --fs_license /fs/license.txt \
                  --t1 /data/sub-002/anat/sub-002_T1w.nii.gz \
                  --sid sub-002 --sd /output \
                  --parallel --3T