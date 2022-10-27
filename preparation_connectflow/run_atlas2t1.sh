#!/bin/bash

tractoflow_output='/home/pabaua/dev_tpil/results/results_tracto'
subject_ID='sub-pl010_ses-v1'
t1_image="${tractoflow_output}/${subject_ID}/Bet_T1/${subject_ID}__t1_bet.nii.gz"
fa_image="${tractoflow_output}/${subject_ID}/DTI_Metrics/${subject_ID}__fa.nii.gz"
template_t1='/home/pabaua/Downloads/fsl-6.0.5.2-sources/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz'
template_fa='/home/pabaua/Downloads/FSL_HCP1065_FA_1mm.nii.gz'
atlas='/home/pabaua/Downloads/BN_Atlas_for_FSL/Brainnetome/BNA-maxprob-thr0-1mm.nii.gz'
output="/home/pabaua/dev_tpil/data/connectflow_data/${subject_ID}"

mkdir ${output}
cd ${output}
mkdir metrics
mkdir tmp
cp ${atlas} ${subject_ID}__labels.nii.gz

antsRegistrationSyN.sh -d 3 -f ${template_t1} -m ${t1_image} -t s -o tmp/${subject_ID}_t12template__
cp tmp/${subject_ID}_t12template__Warped.nii.gz ${subject_ID}__t1.nii.gz

antsRegistrationSyN.sh -d 3 -f ${fa_image} -m ${template_fa} -t s -o tmp/${subject_ID}_template2fa__
cp tmp/${subject_ID}_template2fa__1Warp.nii.gz ${subject_ID}__1Warp.nii.gz
cp tmp/${subject_ID}_template2fa__0GenericAffine.mat ${subject_ID}__0GenericAffine.mat

cp ${tractoflow_output}/${subject_ID}/Resample_DWI/${subject_ID}__dwi_resampled.nii.gz ${subject_ID}__dwi.nii.gz
cp ${tractoflow_output}/${subject_ID}/Eddy_Topup/${subject_ID}__dwi_eddy_corrected.bvec ${subject_ID}__dwi.bvec
cp ${tractoflow_output}/${subject_ID}/Eddy_Topup/${subject_ID}__bval_eddy ${subject_ID}__dwi.bval
cp ${tractoflow_output}/${subject_ID}/PFT_Tracking/${subject_ID}__pft_tracking_prob_wm_seed_0.trk ${subject_ID}__pft_tracking_prob_wm_seed_0.trk
cp ${tractoflow_output}/${subject_ID}/FODF_Metrics/${subject_ID}__fodf.nii.gz ${subject_ID}__fodf.nii.gz
cp ${tractoflow_output}/${subject_ID}/FODF_Metrics/${subject_ID}__peaks.nii.gz ${subject_ID}__peaks.nii.gz
cp ${tractoflow_output}/${subject_ID}/DTI_Metrics/${subject_ID}__fa.nii.gz metrics/${subject_ID}__fa.nii.gz

