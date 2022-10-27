#!/bin/bash
#
# Copies all files and directories into new folder "<DATASET_ROOT_FOLDER>_lowercase"
# and converts all filenames to lowercase in new folder.
# This script should be run beside the dataset root folder. If rename command is not found run:
# 'sudo apt install rename'
#
# Usage:
#   ./lowercase_files <DATASET_ROOT_FOLDER>
#
# Example:
#   ./lowercase_files Data_dMRI
#
# Author: Paul Bautin
###################################################

# Uncomment for full verbose
set -x

# Immediately exit if error
set -e

# Exit if user presses CTRL+C (Linux) or CMD+C (OSX)
trap "echo Caught Keyboard Interrupt within script. Exiting now.; exit" INT

# flag d -> image dimensions
# flag t -> s: rigid + affine + deformable syn (3 stages)
# flag o -> output prefix
atlas='/home/pabaua/dev_tpil/results/results_freesurfer_flow/22-10-18_freesurfer-flow/results_fs/sub-pl007_ses-v1/Generate_Atlases_FS_BN_GL_SF/atlas_brainnetome_v4.nii.gz'
original_t1='/home/pabaua/dev_tpil/results/results_tracto/sub-pl007_ses-v1/Bet_T1/sub-pl007_ses-v1__t1_bet.nii.gz'
ref_img='/home/pabaua/dev_tpil/results/results_tracto/sub-pl007_ses-v1/DTI_Metrics/sub-pl007_ses-v1__fa.nii.gz'
warp='/home/pabaua/dev_tpil/results/results_tracto/sub-pl007_ses-v1/Register_T1/sub-pl007_ses-v1__output1Warp.nii.gz'
affine='/home/pabaua/dev_tpil/results/results_tracto/sub-pl007_ses-v1/Register_T1/sub-pl007_ses-v1__output0GenericAffine.mat'
out_atlas='/home/pabaua/dev_tpil/results/results_freesurfer_flow/out_atlas.nii.gz'
#antsRegistrationSyNQuick.sh -d 3 -f ${ref_img} -m ${original_t1} -t s -o to_reference_
#antsRegistration -d 3 -f ${ref_img} -m ${atlas} -t s -o to_reference_ -n NearestNeighbor -v 1 -c 1e-6
# cannot use output 'to_reference_Warped.nii.gz' because interpolation changes atlas labels -> float 64 instead of integer
antsApplyTransforms -d 3 -i ${atlas} -t ${warp} -t ${affine} -r ${warp} -o ${out_atlas} -n NearestNeighbor


warp='/home/pabaua/tpil_dev/github/tpil_dmri/to_reference_1Warp.nii.gz'
affine='/home/pabaua/tpil_dev/github/tpil_dmri/to_reference_0GenericAffine.mat'
# out_atlas='/home/pabaua/tpil_dev/results/warped_atlas.nii.gz'
out_atlas='warped_atlas.nii.gz'
ref_image_atlas='/home/pabaua/tpil_dev/github/tpil_dmri/to_reference_Warped.nii.gz'
ref_image='/home/pabaua/tpil_dev/results/results_rbx/DCL_PT_PL007_V1/Register_Anat/DCL_PT_PL007_V1__outputWarped.nii.gz'
# ref_image_atlas='/home/pabaua/tpil_dev/results/results/DCL_PT_PL007_V1/Segment_Tissues/DCL_PT_PL007_V1__mask_wm.nii.gz'
# scil_apply_transform_to_image.py ${atlas}
#antsApplyTransforms -d 3 -i ${atlas} -t ${warp} -t ${affine} -r ${ref_image} -o ${out_atlas} -n genericLabel -v 1 --float 0

out_atlas_int='warped_atlas_int.nii.gz'
#scil_image_math.py convert ${out_atlas} ${out_atlas_int} --data_type int16



my_trk='/home/pabaua/tpil_dev/results/results/DCL_PT_PL007_V1/Local_Tracking/DCL_PT_PL007_V1__local_tracking_prob_fa_seeding_fa_mask_seed_0.trk'
# my_trk='/home/pabaua/tpil_dev/results/warped_tractogram.trk'
out_trk='/home/pabaua/tpil_dev/results/DCL_PT_PL007_V1__NAC_mPFC_filtered.trk'
#atlas_warped='/home/pabaua/tpil_dev/github/tpil_dmri/to_reference_Warped.nii.gz'
parameters_NAC_l="--drawn_roi /home/pabaua/tpil_dev/results/mask_mPFC.nii.gz  either_end include"
parameters_NAC_r="--atlas_roi ${out_atlas_int} 224 any include"
parameters_A8m_l="--drawn_roi /home/pabaua/tpil_dev/results/mask_NAC.nii.gz either_end include"
parameters_A8m_r="--atlas_roi ${out_atlas_int} 14 any include"
# 223 -> label number for nucleus accumbens in SBPr (NAC)
#scil_filter_tractogram.py ${my_trk} ${out_trk} ${parameters_NAC_l} ${parameters_NAC_r} ${parameters_A8m_l} ${parameters_A8m_r}
#scil_filter_tractogram.py ${my_trk} ${out_trk} ${parameters_NAC_l} ${parameters_A8m_l}

out_trk_cleaned='/home/pabaua/tpil_dev/results/DCL_PT_PL007_V1__NAC_mPFC_cleaned.trk'
#scil_outlier_rejection.py ${out_trk} ${out_trk_cleaned} --alpha 0.5


out_centroid='/home/pabaua/tpil_dev/results/DCL_PT_PL007_V1__NAC_mPFC_centroid.trk'
#scil_compute_centroid.py ${out_trk_cleaned} ${out_centroid} --nb_points 50



